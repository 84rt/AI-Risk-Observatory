#!/usr/bin/env python3
"""Build a single documents_manifest.json covering the full corpus.

Three sources are merged in priority order (higher priority wins on duplicate PKs):
  1. FR_consolidated  — data/FR_consolidated/metadata.csv
  2. Gap fill 2021    — data/ch_gap_fill_2021/gap_manifest.csv
  3. Gap fill FY22-25 — data/processed/ch-gap-fill-20260330/documents_manifest.json

For gap-fill sources, stub and duplicate detection is applied on the fly:
  - Stubs: markdown files < 50 KB whose header is an NSM/RNS notification wrapper.
  - Duplicates: for each (lei, fiscal_year), prefer the PK in target_manifest.fr_pk,
    fall back to the largest file.  Non-preferred rows are skipped.

Usage:
    python scripts/build_unified_manifest.py [--run-id unified-20260330] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"

FR_CONSOLIDATED_META = DATA_ROOT / "FR_consolidated" / "metadata.csv"
GAP_2021_MANIFEST = DATA_ROOT / "ch_gap_fill_2021" / "gap_manifest.csv"
GAP_2021_DIRECT_MANIFEST = DATA_ROOT / "ch_gap_fill_2021_direct" / "gap_manifest.csv"
GAP_MAIN_MANIFEST = DATA_ROOT / "processed" / "ch-gap-fill-20260330" / "documents_manifest.json"
TARGET_MANIFEST = DATA_ROOT / "reference" / "target_manifest.csv"
PROCESSED_DIR = DATA_ROOT / "processed"

STUB_SIZE_THRESHOLD = 50_000
NSM_MARKERS = ("National Storage Mechanism", "RNS Number", "News Details")
NSM_HEAD_BYTES = 500
ACTIVE_GAP_STATUSES = {"fr_recovered", "ch_processed"}


def parse_args() -> argparse.Namespace:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Build unified corpus documents manifest.")
    parser.add_argument("--run-id", default=f"unified-{today}")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stub detection (same logic as build_gap_fill_manifest.py)
# ---------------------------------------------------------------------------

def _is_stub(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return True
    if size >= STUB_SIZE_THRESHOLD:
        return False
    if size < 5_000:
        return True
    try:
        head = path.read_bytes()[:NSM_HEAD_BYTES].decode("utf-8", errors="replace")
        return any(m in head for m in NSM_MARKERS)
    except OSError:
        return True


# ---------------------------------------------------------------------------
# Target manifest lookup
# ---------------------------------------------------------------------------

def load_target(path: Path) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        result[(row["lei"], row["fiscal_year"])] = row
    return result


# ---------------------------------------------------------------------------
# Source 1: FR_consolidated
# ---------------------------------------------------------------------------

def records_from_fr_consolidated(meta_path: Path, run_id: str) -> list[dict]:
    records = []
    for row in csv.DictReader(meta_path.open(encoding="utf-8")):
        src_path = Path(row["src_path"])
        if not src_path.exists():
            continue
        fiscal_year = row.get("fiscal_year") or row.get("release_year") or ""
        try:
            year = int(fiscal_year)
        except ValueError:
            continue
        records.append({
            "document_id": row["pk"],
            "company_id": row["lei"],
            "company_number": "",
            "company_name": row["company"],
            "ticker": "",
            "lei": row["lei"],
            "cni_sector": row.get("cni_sector_primary") or "",
            "year": year,
            "release_year": int(row.get("release_year") or year),
            "source_format": "fr_markdown",
            "source": "fr_api",
            "fr_pk": row["pk"],
            "markdown_path": str(src_path),
            "market_segment": row.get("market_segment_refined") or "Other",
            "run_id": run_id,
        })
    return records


# ---------------------------------------------------------------------------
# Source 2: Gap fill 2021
# ---------------------------------------------------------------------------

def records_from_gap_2021(
    manifest_path: Path,
    target: dict[tuple[str, str], dict],
    run_id: str,
) -> list[dict]:
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    active = [r for r in rows if r["status"] in ACTIVE_GAP_STATUSES]

    # Resolve markdown paths (stored relative to repo root in this manifest)
    for row in active:
        mp = row.get("markdown_path", "")
        if mp and not Path(mp).is_absolute():
            row["_abs_path"] = REPO_ROOT / mp
        else:
            row["_abs_path"] = Path(mp) if mp else None

    # Mark stubs
    active = [r for r in active if r["_abs_path"] and not _is_stub(r["_abs_path"])]

    # Deduplicate: (lei, fiscal_year) → prefer target_manifest.fr_pk, else largest file
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in active:
        groups[(row["lei"], row["fiscal_year"])].append(row)

    selected = []
    for (lei, yr), group in groups.items():
        if len(group) == 1:
            selected.append(group[0])
            continue
        t = target.get((lei, yr))
        target_pk = t.get("fr_pk") if t else None
        preferred = next((r for r in group if r["pk"] == target_pk), None)
        if preferred is None:
            preferred = max(group, key=lambda r: r["_abs_path"].stat().st_size if r["_abs_path"].exists() else 0)
        selected.append(preferred)

    records = []
    for row in selected:
        lei = row["lei"]
        yr = row["fiscal_year"]
        t = target.get((lei, yr), {})
        try:
            year = int(yr)
        except ValueError:
            continue
        source_format = "fr_markdown" if row["status"] == "fr_recovered" else "mistral_ocr"
        source = "fr_api" if row["status"] == "fr_recovered" else "companies_house"
        records.append({
            "document_id": row["pk"],
            "company_id": lei,
            "company_number": t.get("ch_company_number") or row.get("ch_company_number") or "",
            "company_name": t.get("company_name") or row.get("company_name") or "",
            "ticker": "",
            "lei": lei,
            "cni_sector": t.get("cni_sector_primary") or "",
            "year": year,
            "release_year": 2021,
            "source_format": source_format,
            "source": source,
            "fr_pk": row["pk"],
            "markdown_path": str(row["_abs_path"]),
            "market_segment": t.get("market_segment_refined") or t.get("market_segment") or "Other",
            "run_id": run_id,
        })
    return records


# ---------------------------------------------------------------------------
# Source 3: Gap fill 2021 direct (CH OCR, no FR PK)
# ---------------------------------------------------------------------------

def records_from_gap_2021_direct(
    manifest_path: Path,
    target: dict[tuple[str, str], dict],
    run_id: str,
) -> list[dict]:
    if not manifest_path.exists():
        return []
    records = []
    for row in csv.DictReader(manifest_path.open(encoding="utf-8")):
        if row.get("status") != "ch_processed":
            continue
        mp = row.get("markdown_path", "")
        if not mp:
            continue
        abs_path = Path(mp) if Path(mp).is_absolute() else REPO_ROOT / mp
        if not abs_path.exists():
            continue
        lei = row["lei"]
        yr = row.get("fiscal_year", "")
        try:
            year = int(yr)
        except ValueError:
            continue
        t = target.get((lei, yr), {})
        # Use the compound id as document_id (no FR PK exists)
        doc_id = row["id"]
        records.append({
            "document_id": doc_id,
            "company_id": lei,
            "company_number": t.get("ch_company_number") or row.get("ch_company_number") or "",
            "company_name": t.get("company_name") or row.get("company_name") or "",
            "ticker": "",
            "lei": lei,
            "cni_sector": t.get("cni_sector_primary") or "",
            "year": year,
            "release_year": 2021,
            "source_format": "mistral_ocr",
            "source": "companies_house",
            "fr_pk": doc_id,
            "markdown_path": str(abs_path),
            "market_segment": t.get("market_segment_refined") or t.get("market_segment") or "Other",
            "run_id": run_id,
        })
    return records


# ---------------------------------------------------------------------------
# Source 4: Gap fill FY2022–25 (already built manifest)
# ---------------------------------------------------------------------------

def records_from_gap_main(manifest_path: Path, run_id: str) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for d in manifest.get("documents", []):
        rec = dict(d)
        rec["run_id"] = run_id
        # release_year ≈ fiscal year for annual reports
        rec.setdefault("release_year", rec.get("year"))
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print("Loading target manifest…")
    target = load_target(TARGET_MANIFEST)

    print("Loading FR_consolidated…")
    fr_records = records_from_fr_consolidated(FR_CONSOLIDATED_META, args.run_id)
    print(f"  {len(fr_records)} records")

    print("Loading gap fill 2021…")
    gap2021_records = records_from_gap_2021(GAP_2021_MANIFEST, target, args.run_id)
    print(f"  {len(gap2021_records)} records (after stub/dedup filtering)")

    print("Loading gap fill 2021 direct (CH OCR)…")
    gap_2021_direct_records = records_from_gap_2021_direct(GAP_2021_DIRECT_MANIFEST, target, args.run_id)
    print(f"  {len(gap_2021_direct_records)} records")

    print("Loading gap fill FY2022–25…")
    gap_main_records = records_from_gap_main(GAP_MAIN_MANIFEST, args.run_id)
    print(f"  {len(gap_main_records)} records")

    # Merge — deduplicate by PK (fr_consolidated wins, then gap fills in order)
    seen_pks: set[str] = set()
    all_records: list[dict] = []
    for source_records in [fr_records, gap2021_records, gap_2021_direct_records, gap_main_records]:
        for rec in source_records:
            pk = rec["fr_pk"]
            if pk in seen_pks:
                continue
            seen_pks.add(pk)
            all_records.append(rec)

    # Summary
    from collections import Counter
    by_year = Counter(r["release_year"] for r in all_records)
    by_source = Counter(r["source_format"] for r in all_records)
    print(f"\nTotal records: {len(all_records)}")
    print("By release year:", dict(sorted(by_year.items())))
    print("By source format:", dict(by_source))

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return 0

    output_dir = PROCESSED_DIR / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "documents_manifest.json"

    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "unified",
        "document_count": len(all_records),
        "documents": all_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote → {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
