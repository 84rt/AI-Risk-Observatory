#!/usr/bin/env python3
"""Build FULL_DB: canonical company and filing registry.

Universe source of truth: ch_period_of_accounts.csv (~1,300–1,370 companies/year).
Company metadata (market segment, CNI sector) joined from target_manifest.csv and
ch_coverage.csv where available.

Outputs:
  data/FULL_DB/companies.csv  — one row per LEI (~1,400 companies)
  data/FULL_DB/filings.csv    — one row per (LEI, fiscal_year) for FY2021–2025

Document resolution priority (first match wins for each (lei, fiscal_year)):
  1. FR_consolidated/metadata.csv        — highest quality, already processed
  2. ch_gap_fill_2021/gap_manifest.csv   — FR recovered or OCR (2021 gap fill)
  3. ch_gap_fill_2021_direct             — direct CH OCR (2021 companies not in FR)
  4. ch_gap_fill/gap_manifest.csv        — FR recovered or OCR (FY2022–25 gap fill)

release_year comes from FR_dataset/manifest.csv (authoritative); falls back to
ch_submission_date year from ch_period_of_accounts, then fiscal_year.

Usage:
    python scripts/build_full_db.py [--dry-run] [--years 2021,2022,2023,2024,2025]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"
PIPELINE_ROOT = REPO_ROOT / "pipeline"

CH_PERIOD_OF_ACCOUNTS = DATA_ROOT / "FR_dataset" / "ch_period_of_accounts.csv"
CH_COVERAGE = DATA_ROOT / "FR_dataset" / "ch_coverage.csv"
TARGET_MANIFEST = DATA_ROOT / "reference" / "target_manifest.csv"
FR_DATASET_MANIFEST = DATA_ROOT / "FR_dataset" / "manifest.csv"
FR_CONSOLIDATED_META = DATA_ROOT / "FR_consolidated" / "metadata.csv"

GAP_2021_MANIFEST = DATA_ROOT / "ch_gap_fill_2021" / "gap_manifest.csv"
GAP_2021_DIRECT_MANIFEST = DATA_ROOT / "ch_gap_fill_2021_direct" / "gap_manifest.csv"
GAP_MAIN_MANIFEST = DATA_ROOT / "ch_gap_fill" / "gap_manifest.csv"

CHUNKS_FILE = DATA_ROOT / "processed" / "unified-20260330" / "chunks" / "chunks.jsonl"
DEFINITIVE_1000_MANIFEST = DATA_ROOT / "processed" / "fr-phase1-definitive-main-market-1000" / "documents_manifest.json"

FULL_DB_DIR = DATA_ROOT / "FULL_DB"
FISCAL_YEARS = {"2021", "2022", "2023", "2024", "2025"}

STUB_SIZE_THRESHOLD = 50_000
NSM_MARKERS = ("National Storage Mechanism", "RNS Number", "News Details")
NSM_HEAD_BYTES = 500

ACTIVE_GAP_STATUSES = {"fr_recovered", "ch_processed"}


# ---------------------------------------------------------------------------
# Helpers
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


def _file_size_kb(path_str: str) -> float:
    try:
        return Path(path_str).stat().st_size / 1024
    except OSError:
        return 0.0


def _resolve_path(raw: str) -> Path | None:
    """Resolve a markdown path — may be absolute or relative to REPO_ROOT."""
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


# ---------------------------------------------------------------------------
# Step 1: Load the universe from ch_period_of_accounts (CH source of truth)
# ---------------------------------------------------------------------------

def load_ch_universe(fiscal_years: set[str]) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    """Build universe from ch_period_of_accounts.csv filtered to fiscal_years.

    Returns (all_rows, lookup by (lei, fiscal_year)).
    Each row has: lei, company_number, company_name, fiscal_year,
                  submission_date, made_up_date, filing_type.
    Deduplicates to one row per (lei, fiscal_year): prefers AA filing type,
    then latest submission_date.
    """
    raw = list(csv.DictReader(CH_PERIOD_OF_ACCOUNTS.open(encoding="utf-8")))

    # Keep only target fiscal years
    raw = [r for r in raw if r["fiscal_year"] in fiscal_years]

    # Group by (lei, fiscal_year) — deduplicate
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in raw:
        groups[(r["lei"], r["fiscal_year"])].append(r)

    rows = []
    for (lei, fy), group in groups.items():
        # Prefer AA (annual accounts) over other types, then latest submission
        aa = [r for r in group if r.get("filing_type") == "AA"]
        pool = aa if aa else group
        best = max(pool, key=lambda r: r.get("submission_date", ""))
        rows.append(best)

    lookup = {(r["lei"], r["fiscal_year"]): r for r in rows}
    return rows, lookup


def load_company_metadata() -> dict[str, dict]:
    """LEI -> metadata dict from target_manifest and ch_coverage."""
    meta: dict[str, dict] = {}

    # Base: ch_coverage (market_segment for all companies)
    for row in csv.DictReader(CH_COVERAGE.open(encoding="utf-8")):
        lei = row["lei"]
        meta[lei] = {
            "company_name": row.get("name", ""),
            "ch_company_number": row.get("company_number", ""),
            "market_segment": row.get("market_segment", ""),
            "market_segment_refined": "",
            "cni_sector_primary": "",
            "cni_sectors": "",
            "isic_code": "",
            "isic_name": "",
        }

    # Enrich with target_manifest (richer metadata, last row per LEI wins)
    for row in csv.DictReader(TARGET_MANIFEST.open(encoding="utf-8")):
        lei = row["lei"]
        if lei not in meta:
            meta[lei] = {}
        meta[lei].update({
            "company_name": row.get("company_name") or meta[lei].get("company_name", ""),
            "ch_company_number": row.get("ch_company_number") or meta[lei].get("ch_company_number", ""),
            "market_segment": row.get("market_segment") or meta[lei].get("market_segment", ""),
            "market_segment_refined": row.get("market_segment_refined", ""),
            "cni_sector_primary": row.get("cni_sector_primary", ""),
            "cni_sectors": row.get("cni_sectors", ""),
            "isic_code": row.get("isic_code", ""),
            "isic_name": row.get("isic_name", ""),
        })

    return meta


# ---------------------------------------------------------------------------
# Step 2: Build release_year lookup from FR_dataset
# ---------------------------------------------------------------------------

def load_release_year_map() -> dict[tuple[str, str], str]:
    """(lei, fiscal_year) -> release_year (string) from FR_dataset/manifest.csv."""
    mapping: dict[tuple[str, str], str] = {}
    for row in csv.DictReader(FR_DATASET_MANIFEST.open(encoding="utf-8")):
        key = (row["company__lei"], row["fiscal_year"])
        # If multiple filings exist for the same (lei, fiscal_year), prefer
        # non-ESEF (PDF/markdown tends to be cleaner) but any release_year will do
        if key not in mapping:
            mapping[key] = row["release_year"]
    return mapping


# ---------------------------------------------------------------------------
# Step 3: Build document lookup per source (priority order)
# ---------------------------------------------------------------------------

def _best_gap_doc(group: list[dict]) -> dict:
    """From a group of gap manifest rows for the same (lei, fiscal_year),
    pick the best: prefer fr_recovered over ch_processed, then largest file."""
    if len(group) == 1:
        return group[0]
    fr = [r for r in group if r.get("status") == "fr_recovered"]
    pool = fr if fr else group
    return max(pool, key=lambda r: _file_size_kb(r.get("markdown_path", "")))


def load_fr_consolidated_docs() -> dict[tuple[str, str], dict]:
    """(lei, fiscal_year) -> doc record from FR_consolidated."""
    docs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in csv.DictReader(FR_CONSOLIDATED_META.open(encoding="utf-8")):
        key = (row["lei"], row["fiscal_year"])
        docs[key].append(row)

    result: dict[tuple[str, str], dict] = {}
    for key, group in docs.items():
        # Prefer largest file
        best = max(group, key=lambda r: _file_size_kb(r.get("src_path", "")))
        result[key] = {
            "fr_pk": best["pk"],
            "source": "fr_api",
            "source_format": "fr_markdown",
            "markdown_path": best["src_path"],
            "release_year": best.get("release_year", ""),
        }
    return result


def load_gap_2021_docs() -> dict[tuple[str, str], dict]:
    """(lei, fiscal_year) -> doc record from ch_gap_fill_2021."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in csv.DictReader(GAP_2021_MANIFEST.open(encoding="utf-8")):
        if row.get("status") not in ACTIVE_GAP_STATUSES:
            continue
        p = _resolve_path(row.get("markdown_path", ""))
        if not p or not p.exists():
            continue
        groups[(row["lei"], row["fiscal_year"])].append(row)

    result: dict[tuple[str, str], dict] = {}
    for key, group in groups.items():
        best = _best_gap_doc(group)
        p = _resolve_path(best["markdown_path"])
        source_format = "fr_markdown" if best["status"] == "fr_recovered" else "mistral_ocr"
        result[key] = {
            "fr_pk": best["pk"],
            "source": "fr_api" if best["status"] == "fr_recovered" else "companies_house",
            "source_format": source_format,
            "markdown_path": str(p) if p else best["markdown_path"],
            "release_year": "",
        }
    return result


def load_gap_2021_direct_docs() -> dict[tuple[str, str], dict]:
    """(lei, fiscal_year) -> doc record from ch_gap_fill_2021_direct."""
    if not GAP_2021_DIRECT_MANIFEST.exists():
        return {}
    result: dict[tuple[str, str], dict] = {}
    for row in csv.DictReader(GAP_2021_DIRECT_MANIFEST.open(encoding="utf-8")):
        if row.get("status") != "ch_processed":
            continue
        p = _resolve_path(row.get("markdown_path", ""))
        if not p or not p.exists():
            continue
        key = (row["lei"], row.get("fiscal_year", ""))
        if key in result:
            continue
        result[key] = {
            "fr_pk": row["id"],
            "source": "companies_house",
            "source_format": "mistral_ocr",
            "markdown_path": str(p),
            "release_year": "",
        }
    return result


def load_gap_main_docs() -> dict[tuple[str, str], dict]:
    """(lei, fiscal_year) -> doc record from ch_gap_fill (FY2022-25)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in csv.DictReader(GAP_MAIN_MANIFEST.open(encoding="utf-8")):
        if row.get("status") not in ACTIVE_GAP_STATUSES:
            continue
        p = _resolve_path(row.get("markdown_path", ""))
        if not p or not p.exists():
            continue
        groups[(row["lei"], row["fiscal_year"])].append(row)

    result: dict[tuple[str, str], dict] = {}
    for key, group in groups.items():
        best = _best_gap_doc(group)
        p = _resolve_path(best["markdown_path"])
        source_format = "fr_markdown" if best["status"] == "fr_recovered" else "mistral_ocr"
        result[key] = {
            "fr_pk": best["pk"],
            "source": "fr_api" if best["status"] == "fr_recovered" else "companies_house",
            "source_format": source_format,
            "markdown_path": str(p) if p else best["markdown_path"],
            "release_year": "",
        }
    return result


# ---------------------------------------------------------------------------
# Step 4: Build chunk count lookup from unified chunks
# ---------------------------------------------------------------------------

def load_chunk_counts() -> dict[str, int]:
    """fr_pk (as string) -> chunk count, from unified chunks.jsonl."""
    counts: dict[str, int] = defaultdict(int)
    if not CHUNKS_FILE.exists():
        print(f"  Warning: chunks file not found at {CHUNKS_FILE}")
        return counts
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            counts[str(c["document_id"])] += 1
    return counts


# ---------------------------------------------------------------------------
# Step 5: Build classified_p1 lookup from definitive 1000 run
# ---------------------------------------------------------------------------

def load_classified_p1() -> set[tuple[str, str]]:
    """Set of (lei, fiscal_year) already classified in phase 1."""
    if not DEFINITIVE_1000_MANIFEST.exists():
        return set()
    manifest = json.loads(DEFINITIVE_1000_MANIFEST.read_text(encoding="utf-8"))
    return {(d["lei"], str(d["year"])) for d in manifest["documents"]}


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FULL_DB companies.csv and filings.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    parser.add_argument("--years", default="2021,2022,2023,2024,2025",
                        help="Comma-separated fiscal years to include (default: 2021-2025)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fiscal_years = set(args.years.split(","))

    print("Loading reference data…")
    universe_rows, universe_lookup = load_ch_universe(fiscal_years)
    company_meta = load_company_metadata()
    release_year_map = load_release_year_map()
    classified_p1 = load_classified_p1()
    print(f"  Universe (from CH): {len(universe_rows)} (lei, fiscal_year) pairs")
    from collections import Counter
    fy_dist = Counter(r["fiscal_year"] for r in universe_rows)
    for yr in sorted(fy_dist):
        print(f"    FY{yr}: {fy_dist[yr]}")
    print(f"  Release year map: {len(release_year_map)} entries from FR_dataset")
    print(f"  Already classified (p1): {len(classified_p1)} company-years")

    print("\nLoading document sources…")
    fr_docs = load_fr_consolidated_docs()
    gap2021_docs = load_gap_2021_docs()
    gap2021d_docs = load_gap_2021_direct_docs()
    gap_main_docs = load_gap_main_docs()
    print(f"  FR_consolidated:    {len(fr_docs)} docs")
    print(f"  ch_gap_fill_2021:   {len(gap2021_docs)} docs")
    print(f"  ch_gap_fill_2021d:  {len(gap2021d_docs)} docs")
    print(f"  ch_gap_fill (main): {len(gap_main_docs)} docs")

    print("\nLoading chunk counts…")
    chunk_counts = load_chunk_counts()
    print(f"  {len(chunk_counts)} documents have chunks")

    # ---------- Build filings.csv ----------
    print("\nBuilding filings…")

    filing_rows: list[dict] = []
    status_counts: dict[str, int] = defaultdict(int)

    for row in universe_rows:
        lei = row["lei"]
        fiscal_year = row["fiscal_year"]
        key = (lei, fiscal_year)

        # Resolve release_year
        release_year = release_year_map.get(key, "")
        if not release_year and row.get("submission_date"):
            release_year = row["submission_date"][:4]
        if not release_year:
            release_year = fiscal_year  # last resort: same year

        # Resolve canonical document (priority order)
        doc = (
            fr_docs.get(key)
            or gap2021_docs.get(key)
            or gap2021d_docs.get(key)
            or gap_main_docs.get(key)
        )

        if doc:
            markdown_path = doc["markdown_path"]
            p = Path(markdown_path)
            exists = p.exists()
            stub = _is_stub(p) if exists else True
            size_kb = round(_file_size_kb(markdown_path), 1)
            fr_pk = str(doc["fr_pk"])
            source = doc["source"]
            source_format = doc["source_format"]
            chunk_count = chunk_counts.get(fr_pk, 0)

            if stub:
                status = "stub"
            elif not exists:
                status = "missing"
            else:
                status = "have_markdown"
        else:
            markdown_path = ""
            size_kb = 0.0
            stub = False
            fr_pk = ""
            source = ""
            source_format = ""
            chunk_count = 0
            status = "missing"

        status_counts[status] += 1

        filing_rows.append({
            "lei": lei,
            "fiscal_year": fiscal_year,
            "release_year": release_year,
            "fr_pk": fr_pk,
            "source": source,
            "source_format": source_format,
            "markdown_path": markdown_path,
            "markdown_size_kb": size_kb,
            "is_stub": "yes" if stub else "no",
            "status": status,
            "classified_p1": "yes" if key in classified_p1 else "no",
            "classified_p2": "no",  # placeholder — update after p2 run
            "chunk_count": chunk_count,
        })

    print("\nFiling status breakdown:")
    for status, count in sorted(status_counts.items()):
        pct = 100 * count / len(filing_rows)
        print(f"  {status:<20} {count:>5}  ({pct:.1f}%)")

    from collections import Counter
    release_year_dist = Counter(r["release_year"] for r in filing_rows if r["status"] == "have_markdown")
    print("\nFilings with markdown by release_year:")
    for yr in sorted(release_year_dist):
        print(f"  {yr}: {release_year_dist[yr]}")

    classified_count = sum(1 for r in filing_rows if r["classified_p1"] == "yes")
    ready_to_classify = sum(
        1 for r in filing_rows
        if r["status"] == "have_markdown" and r["classified_p1"] == "no"
    )
    print(f"\nAlready classified (p1): {classified_count}")
    print(f"Ready to classify (have_markdown, not yet p1): {ready_to_classify}")
    print(f"Missing documents: {status_counts.get('missing', 0)}")

    # ---------- Build companies.csv ----------
    seen_leis: set[str] = set()
    company_rows: list[dict] = []
    for row in universe_rows:
        lei = row["lei"]
        if lei in seen_leis:
            continue
        seen_leis.add(lei)
        m = company_meta.get(lei, {})
        company_rows.append({
            "lei": lei,
            "company_name": m.get("company_name") or row.get("name", ""),
            "ch_company_number": m.get("ch_company_number") or row.get("ch_company_number", ""),
            "market_segment": m.get("market_segment", ""),
            "market_segment_refined": m.get("market_segment_refined", ""),
            "cni_sector_primary": m.get("cni_sector_primary", ""),
            "cni_sectors": m.get("cni_sectors", ""),
            "isic_code": m.get("isic_code", ""),
            "isic_name": m.get("isic_name", ""),
        })

    print(f"\nCompanies: {len(company_rows)}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return 0

    # ---------- Write ----------
    FULL_DB_DIR.mkdir(parents=True, exist_ok=True)

    companies_path = FULL_DB_DIR / "companies.csv"
    filings_path = FULL_DB_DIR / "filings.csv"

    company_fields = ["lei", "company_name", "ch_company_number", "market_segment",
                      "market_segment_refined", "cni_sector_primary", "cni_sectors",
                      "isic_code", "isic_name"]
    filing_fields = ["lei", "fiscal_year", "release_year", "fr_pk", "source",
                     "source_format", "markdown_path", "markdown_size_kb", "is_stub",
                     "status", "classified_p1", "classified_p2", "chunk_count"]

    with companies_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=company_fields)
        w.writeheader()
        w.writerows(company_rows)

    with filings_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=filing_fields)
        w.writeheader()
        w.writerows(filing_rows)

    print(f"\nWrote → {companies_path}  ({len(company_rows)} rows)")
    print(f"Wrote → {filings_path}  ({len(filing_rows)} rows)")
    print(f"Done at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
