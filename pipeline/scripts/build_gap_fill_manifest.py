#!/usr/bin/env python3
"""Build a documents_manifest.json for the CH gap-fill run.

Steps performed:
  1. Mark stub files (markdown on disk but too small to be a real report) as
     status=stub in gap_manifest.csv.
  2. Deduplicate: for each (lei, fiscal_year) with multiple recovered PKs,
     prefer the PK matching target_manifest.fr_pk; fall back to the largest
     markdown file.  Tag non-preferred duplicates as status=duplicate.
  3. Emit a documents_manifest.json in the standard chunk_markdown.py format
     for the surviving rows (fr_recovered + ch_processed, excluding stubs and
     duplicates).

Usage:
    python scripts/build_gap_fill_manifest.py [--run-id ch-gap-fill-20260330] [--dry-run]
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
GAP_MANIFEST = DATA_ROOT / "ch_gap_fill" / "gap_manifest.csv"
TARGET_MANIFEST = DATA_ROOT / "reference" / "target_manifest.csv"
FR_RECOVERED_DIR = DATA_ROOT / "ch_gap_fill" / "fr_recovered" / "markdown"
CH_PROCESSED_DIR = DATA_ROOT / "ch_gap_fill" / "ch_processed" / "markdown"
PROCESSED_DIR = DATA_ROOT / "processed"

STUB_SIZE_THRESHOLD = 50_000  # bytes — files below this are treated as stubs
# Files above this threshold may still be stubs if they are NSM/RNS notifications,
# but large files that merely *mention* NSM as one section are real reports and kept.
NSM_MARKERS = ("National Storage Mechanism", "RNS Number", "News Details")
NSM_HEAD_BYTES = 500  # only check the start of the file for NSM markers
ACTIVE_STATUSES = {"fr_recovered", "ch_processed"}
TERMINAL_STATUSES = {"fr_recovered", "ch_processed", "fr_pending", "error", "stub", "duplicate"}

MANIFEST_FIELDS = [
    "pk",
    "lei",
    "company_name",
    "fiscal_year",
    "filing_type",
    "status",
    "fr_processing_status",
    "ch_company_number",
    "ch_filing_date",
    "pdf_path",
    "markdown_path",
    "batch_job_id",
    "error",
]


def parse_args() -> argparse.Namespace:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Build gap-fill documents manifest.")
    parser.add_argument(
        "--gap-manifest",
        type=Path,
        default=GAP_MANIFEST,
        help=f"Path to gap_manifest.csv (default: {GAP_MANIFEST})",
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=TARGET_MANIFEST,
        help=f"Path to target_manifest.csv (default: {TARGET_MANIFEST})",
    )
    parser.add_argument(
        "--run-id",
        default=f"ch-gap-fill-{today}",
        help="Run ID for the output directory under data/processed/ (default: ch-gap-fill-<today>)",
    )
    parser.add_argument(
        "--stub-threshold",
        type=int,
        default=STUB_SIZE_THRESHOLD,
        help=f"Markdown files smaller than this (bytes) are treated as stubs (default: {STUB_SIZE_THRESHOLD})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing any files.",
    )
    return parser.parse_args()


def _markdown_path(row: dict) -> Path | None:
    p = row.get("markdown_path", "")
    return Path(p) if p else None


def _file_size(row: dict) -> int:
    p = _markdown_path(row)
    if p and p.exists():
        return p.stat().st_size
    return 0


def load_gap_manifest(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def save_gap_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_target_manifest(path: Path) -> dict[tuple[str, str], dict]:
    """Return dict keyed by (lei, fiscal_year)."""
    mapping: dict[tuple[str, str], dict] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        key = (row["lei"], row["fiscal_year"])
        mapping[key] = row
    return mapping


def _is_nsm_stub(path: Path) -> bool:
    """Return True if the file starts with an NSM/RNS notification header.

    These are regulatory announcement wrappers that link to the real report
    externally.  They are not the annual report itself.
    """
    try:
        head = path.read_bytes()[:NSM_HEAD_BYTES].decode("utf-8", errors="replace")
        return any(marker in head for marker in NSM_MARKERS)
    except OSError:
        return False


def mark_stubs(rows: list[dict], threshold: int) -> tuple[list[dict], int]:
    """Mark fr_recovered/ch_processed rows as stub if:
      - the markdown file is smaller than threshold bytes, OR
      - the file starts with an NSM/RNS notification header (regardless of size).

    Large real reports may mention NSM in passing — we only flag the header case
    when the file is also below the threshold, catching all notification stubs
    while keeping full reports that happen to include an NSM disclosure section.
    """
    count = 0
    for row in rows:
        if row["status"] not in ACTIVE_STATUSES:
            continue
        p = _markdown_path(row)
        size = _file_size(row)
        is_stub = size < threshold and (size < 5_000 or (p is not None and _is_nsm_stub(p)))
        if is_stub:
            row["status"] = "stub"
            row["error"] = f"stub: file too small or NSM/RNS wrapper ({size} bytes)"
            count += 1
    return rows, count


def deduplicate(
    rows: list[dict],
    target: dict[tuple[str, str], dict],
) -> tuple[list[dict], int]:
    """For each (lei, fiscal_year) with multiple active PKs, keep one canonical row.

    Preference order:
      1. PK matching target_manifest.fr_pk for that (lei, year)
      2. Largest markdown file on disk
    """
    # Group active rows by (lei, fiscal_year)
    active_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["status"] in ACTIVE_STATUSES:
            active_by_key[(row["lei"], row["fiscal_year"])].append(row)

    duplicate_count = 0
    for (lei, yr), group in active_by_key.items():
        if len(group) <= 1:
            continue

        # Determine canonical PK
        t = target.get((lei, yr))
        target_pk = t.get("fr_pk") if t else None

        preferred = None
        if target_pk:
            preferred = next((r for r in group if r["pk"] == target_pk), None)

        if preferred is None:
            # Fall back to largest file
            preferred = max(group, key=_file_size)

        for row in group:
            if row is not preferred:
                row["status"] = "duplicate"
                row["error"] = f"duplicate: canonical_pk={preferred['pk']}"
                duplicate_count += 1

    return rows, duplicate_count


def build_manifest_record(
    row: dict,
    target: dict[tuple[str, str], dict],
    run_id: str,
) -> dict:
    """Convert a gap manifest row to a documents_manifest.json record."""
    lei = row["lei"]
    yr = row["fiscal_year"]
    t = target.get((lei, yr), {})

    source_format = "fr_markdown" if row["status"] == "fr_recovered" else "mistral_ocr"
    source = "fr_api" if row["status"] == "fr_recovered" else "companies_house"
    market_segment = t.get("market_segment_refined") or t.get("market_segment") or "Other"
    cni_sector = t.get("cni_sector_primary") or ""
    company_name = t.get("company_name") or row.get("company_name") or ""
    ch_company_number = t.get("ch_company_number") or row.get("ch_company_number") or ""

    return {
        "document_id": row["pk"],
        "company_id": lei,
        "company_number": ch_company_number,
        "company_name": company_name,
        "ticker": "",
        "lei": lei,
        "cni_sector": cni_sector,
        "year": int(yr),
        "source_format": source_format,
        "source": source,
        "fr_pk": row["pk"],
        "markdown_path": str(_markdown_path(row)),
        "market_segment": market_segment,
        "run_id": run_id,
    }


def main() -> int:
    args = parse_args()

    print("Loading manifests…")
    rows = load_gap_manifest(args.gap_manifest)
    target = load_target_manifest(args.target_manifest)
    print(f"  gap_manifest rows: {len(rows)}")
    print(f"  target_manifest entries: {len(target)}")

    # Step 1: mark stubs
    rows, stub_count = mark_stubs(rows, args.stub_threshold)
    print(f"\nStubs marked: {stub_count}")

    # Step 2: deduplicate
    rows, dup_count = deduplicate(rows, target)
    print(f"Duplicates marked: {dup_count}")

    # Final counts
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[row["status"]] += 1
    print("\nStatus breakdown after deduplication:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Rows to include in manifest
    include_rows = [r for r in rows if r["status"] in ACTIVE_STATUSES]
    print(f"\nRows included in documents_manifest.json: {len(include_rows)}")
    by_source = defaultdict(int)
    for r in include_rows:
        by_source[r["status"]] += 1
    for k, v in sorted(by_source.items()):
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return 0

    # Step 3: build documents manifest
    output_dir = PROCESSED_DIR / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "documents_manifest.json"

    records = [build_manifest_record(r, target, args.run_id) for r in include_rows]
    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "ch_gap_fill",
        "document_count": len(records),
        "documents": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote documents_manifest.json → {manifest_path}")

    # Step 4: save updated gap manifest
    save_gap_manifest(args.gap_manifest, rows)
    print(f"Updated gap_manifest.csv → {args.gap_manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
