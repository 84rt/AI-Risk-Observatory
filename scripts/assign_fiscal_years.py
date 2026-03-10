#!/usr/bin/env python3
"""Batch fiscal-year assignment correction for existing manifest entries.

Runs a multi-layer pipeline per filing to detect and correct mis-assigned
fiscal years in data/FR_dataset/manifest.csv.

Layers (in priority order):
  Layer 3 — Markdown ground truth (highest confidence)
  Layer 1 — Title period-end date
  Layer 2 — Publication-before-period-end sanity check
  Layer 4 — Intra-company clustering (groups HTML + ESEF pairs)
  Fallback — Q1 heuristic / release year

Outputs:
  data/reference/fy_corrections.csv  — detected changes (HIGH/MEDIUM confidence)
  data/reference/fy_qa_review.csv    — anomalies for human review

Usage:
  python scripts/assign_fiscal_years.py            # detect only, print summary
  python scripts/assign_fiscal_years.py --apply    # write corrections to manifest.csv
  python scripts/assign_fiscal_years.py --limit 20 # first 20 companies (debug)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR       = REPO_ROOT / "data" / "FR_dataset"
MANIFEST_CSV   = DATA_DIR / "manifest.csv"
PS_CSV         = DATA_DIR / "processing_status.csv"
MD_DIR         = DATA_DIR / "markdown"

CACHE_DIRS = [
    REPO_ROOT / "data" / "FR_clean"               / "markdown",
    REPO_ROOT / "data" / "FR-UK-2021-2023-test-2" / "markdown",
    REPO_ROOT / "data" / "FR_2026-02-05"          / "markdown",
    MD_DIR,
]

REF_DIR          = REPO_ROOT / "data" / "reference"
CORRECTIONS_CSV  = REF_DIR / "fy_corrections.csv"
QA_REVIEW_CSV    = REF_DIR / "fy_qa_review.csv"

TARGET_YEARS = {"2021", "2022", "2023", "2024", "2025", "2026"}

CORRECTIONS_FIELDS = ["pk", "lei", "name", "manifest_fy", "detected_fy",
                      "confidence", "signal_source"]
QA_FIELDS          = ["pk", "lei", "name", "manifest_fy", "detected_fy",
                      "confidence", "anomaly_type", "notes"]

MARKDOWN_LINES = 100  # lines to read from each .md file

# ── Date-extraction helpers (mirrors refresh_fr_status.py) ───────────────────

_MONTH_NAMES = (
    "January|February|March|April|May|June|"
    "July|August|September|October|November|December"
)
_MONTH_ABBR = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

_PERIOD_END_PATTERNS = [
    re.compile(
        r'\b(\d{1,2})\s+(' + _MONTH_NAMES + r')\s+(20\d{2})\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(\d{1,2})\s+(' + _MONTH_ABBR + r')\.?\s+(20\d{2})\b',
        re.IGNORECASE,
    ),
]

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Patterns used in markdown to identify the fiscal year period
_MD_YEAR_PATTERNS = [
    re.compile(
        r'(?:for the year ended|year ended|financial year ended|'
        r'financial statements for the year)\s+'
        r'(?:\d{1,2}\s+)?(?:' + _MONTH_NAMES + r'|' + _MONTH_ABBR + r')\.?\s+(20\d{2})',
        re.IGNORECASE,
    ),
    re.compile(
        r'(?:for the year ended|year ended|financial year ended)\s+'
        r'(\d{1,2})\s+(?:' + _MONTH_NAMES + r'|' + _MONTH_ABBR + r')\.?\s+(20\d{2})',
        re.IGNORECASE,
    ),
    # Standalone header like "31 December 2022" on its own line
    re.compile(
        r'^(\d{1,2})\s+(?:' + _MONTH_NAMES + r'|' + _MONTH_ABBR + r')\.?\s+(20\d{2})\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
]


def extract_period_end_date(title: str) -> Optional[tuple[int, int, int]]:
    """Extract (year, month, day) from a title like '31 August 2023'."""
    for pat in _PERIOD_END_PATTERNS:
        m = pat.search(title)
        if m:
            day = int(m.group(1))
            month = _MONTH_MAP.get(m.group(2).lower())
            year = int(m.group(3))
            if month and 1 <= day <= 31 and 2015 <= year <= 2030:
                return (year, month, day)
    return None


def extract_year_from_markdown(pk: str) -> Optional[int]:
    """Read first MARKDOWN_LINES lines of {pk}.md and extract fiscal year."""
    for cache_dir in CACHE_DIRS:
        md_path = cache_dir / f"{pk}.md"
        if md_path.exists():
            try:
                with open(md_path, encoding="utf-8", errors="replace") as f:
                    lines = [next(f) for _ in range(MARKDOWN_LINES)]
                text = "".join(lines)
            except StopIteration:
                text = open(md_path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue

            for pat in _MD_YEAR_PATTERNS:
                for m in pat.finditer(text):
                    # Last group is always the 4-digit year
                    year_str = m.group(m.lastindex)
                    try:
                        year = int(year_str)
                        if 2015 <= year <= 2030:
                            return year
                    except (ValueError, TypeError):
                        pass
    return None


# ── Release-date helpers ──────────────────────────────────────────────────────

def parse_release_date(release_dt: str) -> Optional[tuple[int, int, int]]:
    """Parse YYYY-MM-DD prefix from release_datetime string."""
    try:
        parts = release_dt[:10].split("-")
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        return None


def days_between(dt1: str, dt2: str) -> Optional[int]:
    """Return absolute day-delta between two ISO datetime strings."""
    try:
        from datetime import date
        d1 = date.fromisoformat(dt1[:10])
        d2 = date.fromisoformat(dt2[:10])
        return abs((d2 - d1).days)
    except (ValueError, TypeError):
        return None


# ── Per-filing analysis ───────────────────────────────────────────────────────

def analyse_filing(row: dict, md_status: str) -> dict:
    """Return analysis dict for a single manifest row.

    Keys: detected_fy, confidence, signal_source, anomaly_type, notes
    """
    pk         = row["pk"]
    title      = row.get("title", "")
    release_dt = row.get("release_datetime", "")
    manifest_fy = row.get("fiscal_year", "")

    result = {
        "detected_fy":  None,
        "confidence":   "LOW",
        "signal_source": "release_year_fallback",
        "anomaly_type": "",
        "notes":        "",
    }

    # ── Layer 3: markdown ground truth ────────────────────────────────────────
    if md_status in ("cached", "fetched"):
        md_year = extract_year_from_markdown(pk)
        if md_year is not None:
            result["detected_fy"]   = str(md_year)
            result["confidence"]    = "HIGH"
            result["signal_source"] = "markdown_ground_truth"
            return result

    # ── Layers 1 & 2: title period-end date ──────────────────────────────────
    period_end = extract_period_end_date(title)
    if period_end is not None:
        ped_year, ped_month, ped_day = period_end
        release_parts = parse_release_date(release_dt)
        pub_before_end = False
        if release_parts:
            pub_before_end = release_parts < (ped_year, ped_month, ped_day)

        if pub_before_end:
            result["detected_fy"]   = str(ped_year - 1)
            result["confidence"]    = "HIGH"
            result["signal_source"] = "title_period_end_date_sanity_checked"
            result["anomaly_type"]  = "ANOMALY_PUB_BEFORE_END"
            result["notes"]         = (
                f"Published {release_dt[:10]} before period end "
                f"{ped_day:02d}/{ped_month:02d}/{ped_year}"
            )
        else:
            result["detected_fy"]   = str(ped_year)
            result["confidence"]    = "HIGH"
            result["signal_source"] = "title_period_end_date"
        return result

    # ── Fallback: plain year in title ─────────────────────────────────────────
    years = [int(y) for y in re.findall(r'\b(20[12]\d)\b', title)
             if 2015 <= int(y) <= 2030]
    if years:
        result["detected_fy"]   = str(max(years))
        result["confidence"]    = "MEDIUM"
        result["signal_source"] = "title_plain_year"
        return result

    # ── Fallback: Q1 heuristic ────────────────────────────────────────────────
    release_yr = release_dt[:4] if release_dt else ""
    try:
        release_month = int(release_dt[5:7])
    except (ValueError, IndexError):
        release_month = 12

    if release_yr:
        if release_month <= 4:
            result["detected_fy"]   = str(int(release_yr) - 1)
            result["confidence"]    = "MEDIUM"
            result["signal_source"] = "q1_heuristic"
        else:
            result["detected_fy"]   = release_yr
            result["confidence"]    = "LOW"
            result["signal_source"] = "release_year_fallback"

    return result


# ── Clustering ────────────────────────────────────────────────────────────────

def apply_clustering(filings: list[dict]) -> list[dict]:
    """Group filings within 90 days of each other into clusters.

    Propagates the highest-confidence fiscal year within each cluster
    to all members. Flags ANOMALY_CLUSTER if two clusters share a year
    but disagree.

    filings: list of dicts with keys: pk, release_datetime, detected_fy,
             confidence, signal_source, anomaly_type, notes
    Returns the same list with potentially updated values.
    """
    if not filings:
        return filings

    # Sort by release_datetime
    sorted_f = sorted(filings, key=lambda x: x.get("release_datetime", ""))

    # Build clusters: consecutive filings within 90 days
    clusters: list[list[dict]] = []
    current_cluster: list[dict] = [sorted_f[0]]
    for fil in sorted_f[1:]:
        delta = days_between(current_cluster[-1]["release_datetime"],
                             fil["release_datetime"])
        if delta is not None and delta < 90:
            current_cluster.append(fil)
        else:
            clusters.append(current_cluster)
            current_cluster = [fil]
    clusters.append(current_cluster)

    _conf_rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}

    for cluster in clusters:
        # Find best-confidence signal in cluster
        best = max(cluster, key=lambda x: _conf_rank.get(x["confidence"], 0))
        cluster_fy = best["detected_fy"]
        cluster_conf = best["confidence"]
        cluster_src = best["signal_source"]

        # Check for intra-cluster disagreement
        fy_set = {x["detected_fy"] for x in cluster if x["detected_fy"]}
        if len(fy_set) > 1:
            for fil in cluster:
                fil["anomaly_type"] = "ANOMALY_CLUSTER"
                fil["notes"] = f"Cluster year conflict: {sorted(fy_set)}"
        else:
            # Propagate best signal to all cluster members
            for fil in cluster:
                if _conf_rank.get(fil["confidence"], 0) < _conf_rank.get(cluster_conf, 0):
                    fil["detected_fy"]    = cluster_fy
                    fil["confidence"]     = cluster_conf
                    fil["signal_source"]  = f"cluster_propagated({cluster_src})"

    return filings


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    manifest_rows: list[dict],
    md_status_map: dict[str, str],
    limit: Optional[int] = None,
) -> tuple[list[dict], list[dict]]:
    """Run fiscal-year assignment pipeline.

    Returns (corrections, qa_items).
    """
    # Group by company (LEI)
    by_lei: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        by_lei[row["company__lei"]].append(row)

    corrections: list[dict] = []
    qa_items:    list[dict] = []

    leis = sorted(by_lei.keys())
    if limit:
        leis = leis[:limit]

    for lei in leis:
        company_rows = by_lei[lei]
        name = company_rows[0].get("company__name", lei)

        # Analyse each filing individually
        analysed: list[dict] = []
        for row in company_rows:
            pk = row["pk"]
            md_status = md_status_map.get(pk, "")
            analysis = analyse_filing(row, md_status)
            analysed.append({
                "pk":              pk,
                "lei":             lei,
                "name":            name,
                "manifest_fy":     row.get("fiscal_year", ""),
                "release_datetime": row.get("release_datetime", ""),
                **analysis,
            })

        # Layer 4: clustering
        analysed = apply_clustering(analysed)

        for item in analysed:
            manifest_fy  = item["manifest_fy"]
            detected_fy  = item["detected_fy"]
            confidence   = item["confidence"]
            anomaly_type = item["anomaly_type"]

            if anomaly_type:
                qa_items.append({
                    "pk":          item["pk"],
                    "lei":         lei,
                    "name":        name,
                    "manifest_fy": manifest_fy,
                    "detected_fy": detected_fy,
                    "confidence":  confidence,
                    "anomaly_type": anomaly_type,
                    "notes":       item["notes"],
                })
            elif detected_fy and detected_fy != manifest_fy:
                if confidence in ("HIGH", "MEDIUM"):
                    corrections.append({
                        "pk":            item["pk"],
                        "lei":           lei,
                        "name":          name,
                        "manifest_fy":   manifest_fy,
                        "detected_fy":   detected_fy,
                        "confidence":    confidence,
                        "signal_source": item["signal_source"],
                    })

    return corrections, qa_items


# ── Apply corrections ─────────────────────────────────────────────────────────

def apply_corrections(corrections: list[dict]) -> int:
    """Write detected_fy back to manifest.csv for HIGH/MEDIUM confidence rows."""
    if not corrections:
        return 0

    rows = []
    with open(MANIFEST_CSV) as f:
        rows = list(csv.DictReader(f))

    fix_map = {c["pk"]: c["detected_fy"] for c in corrections}
    corrected = 0
    for row in rows:
        if row["pk"] in fix_map:
            old_fy = row["fiscal_year"]
            row["fiscal_year"] = fix_map[row["pk"]]
            print(f"  Correcting pk={row['pk']} ({row['company__name']}): "
                  f"FY{old_fy} → FY{row['fiscal_year']}")
            corrected += 1

    if corrected:
        fieldnames = list(rows[0].keys()) if rows else []
        with open(MANIFEST_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Applied {corrected} corrections to manifest.csv")

    return corrected


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply",  action="store_true",
                   help="Write corrections back to manifest.csv")
    p.add_argument("--limit",  type=int, default=None,
                   help="Process only first N companies (debug)")
    args = p.parse_args()

    REF_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    print(f"Loading {MANIFEST_CSV.relative_to(REPO_ROOT)} ...", end=" ", flush=True)
    with open(MANIFEST_CSV) as f:
        manifest_rows = list(csv.DictReader(f))
    print(f"{len(manifest_rows):,} rows")

    # Load processing_status for md_status lookup
    md_status_map: dict[str, str] = {}
    if PS_CSV.exists():
        print(f"Loading {PS_CSV.relative_to(REPO_ROOT)} ...", end=" ", flush=True)
        with open(PS_CSV) as f:
            for row in csv.DictReader(f):
                md_status_map[row["pk"]] = row.get("md_status", "")
        print(f"{len(md_status_map):,} entries")

    # Run pipeline
    print("\nRunning pipeline...")
    corrections, qa_items = run_pipeline(manifest_rows, md_status_map, limit=args.limit)

    # Write outputs
    with open(CORRECTIONS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CORRECTIONS_FIELDS)
        w.writeheader()
        w.writerows(corrections)

    with open(QA_REVIEW_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QA_FIELDS)
        w.writeheader()
        w.writerows(qa_items)

    # Summary
    high_corrections  = [c for c in corrections if c["confidence"] == "HIGH"]
    med_corrections   = [c for c in corrections if c["confidence"] == "MEDIUM"]
    companies_touched = len({c["lei"] for c in corrections})

    print("\n── Summary ────────────────────────────────────────────────────")
    print(f"  Companies analysed:   {args.limit or len({r['company__lei'] for r in manifest_rows})}")
    print(f"  Corrections detected: {len(corrections):,}  "
          f"(HIGH={len(high_corrections)}, MEDIUM={len(med_corrections)})")
    print(f"  Companies affected:   {companies_touched}")
    print(f"  QA review items:      {len(qa_items):,}")
    print(f"\n  Output: {CORRECTIONS_CSV.relative_to(REPO_ROOT)}")
    print(f"  Output: {QA_REVIEW_CSV.relative_to(REPO_ROOT)}")

    if corrections:
        print("\n── Sample corrections (first 10) ──────────────────────────────")
        for c in corrections[:10]:
            print(f"  pk={c['pk']}  {c['name'][:35]:<35}  "
                  f"FY{c['manifest_fy']} → FY{c['detected_fy']}  "
                  f"[{c['confidence']}] {c['signal_source']}")

    if args.apply:
        print("\n── Applying corrections ────────────────────────────────────────")
        apply_corrections(corrections)
    else:
        print("\nRun with --apply to write corrections to manifest.csv")


if __name__ == "__main__":
    main()
