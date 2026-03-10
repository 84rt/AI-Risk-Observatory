#!/usr/bin/env python3
"""Re-check FR markdown status for stale entries in target_manifest.csv.

Three modes (can be combined):

1. Refresh stale pks  (default, always runs)
   Stale = fr_status in (fr_no_status, fr_pending, fr_failed, fr_skipped).
   All have a known fr_pk → hit /filings/{pk}/markdown/ directly.

2. Check not_in_fr companies  (--check-not-in-fr)
   Query /filings/?lei=... for companies with NO FR entry at all.
   Any new filings found are added to manifest.csv and their markdown fetched.

3. Check year-gaps  (--check-year-gaps)
   Query /filings/?lei=... for companies that have SOME FR entries but are
   missing specific years (not_in_fr alongside other fr_status values).
   Fills holes caused by pagination gaps or late-added filings in FR.

Flow (stale-pk refresh)
-----------------------
1. Load target_manifest.csv → collect stale pks.
2. Check local cache dirs first (free, no API call).
3. Hit /filings/{pk}/markdown/ for the rest.
   200 → save markdown; 404 → read processing_status from body.
4. Merge into phase3_checkpoint.json (overwrites stale entries).
5. Rebuild processing_status.csv.

Flow (not_in_fr check, --check-not-in-fr)
------------------------------------------
1. Collect unique LEIs where ALL years are not_in_fr.
2. GET /filings/?lei={lei}&types=10-K,10-K-ESEF for each.
3. New filings (pk not already in manifest.csv) are appended to manifest.csv
   and their pk added to phase2_checkpoint.json.
4. Markdown is fetched for new pks and checkpoint updated.
5. processing_status.csv and target_manifest.csv are rebuilt.

Flow (year-gap check, --check-year-gaps)
-----------------------------------------
1. Collect LEIs that have SOME FR rows but also some not_in_fr year-gaps.
2. GET /filings/?lei={lei}&types=10-K,10-K-ESEF for each.
3. For each gap year where a matching filing is found, append to manifest.csv.
4. Markdown is fetched for new pks and checkpoint updated.

Outputs
-------
  data/FR_dataset/markdown/{pk}.md       (new markdown files)
  data/FR_dataset/phase2_checkpoint.json (extended with new findings)
  data/FR_dataset/phase3_checkpoint.json (updated in-place)
  data/FR_dataset/manifest.csv           (extended with new filings)
  data/FR_dataset/processing_status.csv  (rebuilt)

Usage
-----
  python scripts/refresh_fr_status.py                          # stale-pk refresh only
  python scripts/refresh_fr_status.py --check-not-in-fr       # + companies absent from FR
  python scripts/refresh_fr_status.py --check-year-gaps       # + year-gap holes
  python scripts/refresh_fr_status.py --dry-run               # cache check, no API calls
  python scripts/refresh_fr_status.py --rebuild-manifest      # also rebuild target_manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env.local", override=True)

API_BASE = "https://api.financialreports.eu"
API_KEY  = os.environ.get("FR_API_KEY", "")
HEADERS  = {"x-api-key": API_KEY, "Accept": "application/json"}

DATA_DIR         = REPO_ROOT / "data" / "FR_dataset"
MANIFEST_CSV     = DATA_DIR / "manifest.csv"
PS_CSV           = DATA_DIR / "processing_status.csv"
P2_CHECKPOINT    = DATA_DIR / "phase2_checkpoint.json"
P3_CHECKPOINT    = DATA_DIR / "phase3_checkpoint.json"
MD_DIR           = DATA_DIR / "markdown"
TARGET_MANIFEST  = REPO_ROOT / "data" / "reference" / "target_manifest.csv"

CACHE_DIRS = [
    REPO_ROOT / "data" / "FR_clean"               / "markdown",
    REPO_ROOT / "data" / "FR-UK-2021-2023-test-2" / "markdown",
    REPO_ROOT / "data" / "FR_2026-02-05"          / "markdown",
    MD_DIR,
]

STALE_STATUSES = {"fr_no_status", "fr_pending", "fr_failed", "fr_skipped"}
TARGET_YEARS   = {"2021", "2022", "2023", "2024", "2025", "2026"}
MIN_YEAR       = "2021"
RATE_LIMIT_SEC = 0.02

MANIFEST_FIELDS = [
    "pk", "company__lei", "company__name", "market_segment",
    "fiscal_year", "release_year", "release_datetime", "title",
    "filing_type__code", "filing_type__name", "is_esef", "candidates_count",
]
PS_FIELDS = [
    "pk", "company__lei", "company__name", "market_segment",
    "fiscal_year", "release_year", "filing_type__code", "is_esef",
    "md_status", "md_size", "md_detail",
]


# ── Shared helpers ────────────────────────────────────────────────────────────

def api_get(path: str, params: Optional[dict] = None,
            retries: int = 3, backoff: float = 5.0) -> requests.Response:
    time.sleep(RATE_LIMIT_SEC)
    last_exc: Exception = RuntimeError("no attempts")
    for attempt in range(1, retries + 1):
        try:
            return requests.get(
                f"{API_BASE}{path}",
                headers=HEADERS,
                params=params or {},
                timeout=60,
            )
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            wait = backoff * attempt
            print(f"    [retry {attempt}/{retries}] {exc.__class__.__name__} — waiting {wait:.0f}s")
            time.sleep(wait)
    raise last_exc


def find_in_cache(pk: str) -> Optional[Path]:
    for d in CACHE_DIRS:
        p = d / f"{pk}.md"
        if p.exists():
            return p
    return None


def extract_markdown_text(resp: requests.Response) -> str:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "json" not in ct:
        return resp.text
    body = resp.json()
    if isinstance(body, dict):
        for key in ("markdown", "content", "text"):
            v = body.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(body, ensure_ascii=False, indent=2)
    return str(body)


def extract_processing_status(resp: requests.Response) -> tuple[str, str]:
    try:
        body = resp.json()
    except ValueError:
        return "unknown", ""
    if not isinstance(body, dict):
        return "unknown", ""
    status = str(body.get("processing_status") or "").strip() or "not_found"
    detail = str(body.get("detail") or "").strip()
    return status, detail


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


def extract_period_end_date(title: str) -> Optional[tuple[int, int, int]]:
    """Extract (year, month, day) from a filing title like '31 August 2023'.

    Returns None if no date is found.
    """
    for pat in _PERIOD_END_PATTERNS:
        m = pat.search(title)
        if m:
            day = int(m.group(1))
            month = _MONTH_MAP.get(m.group(2).lower())
            year = int(m.group(3))
            if month and 1 <= day <= 31 and 2015 <= year <= 2030:
                return (year, month, day)
    return None


def extract_fiscal_year(title: str) -> Optional[int]:
    m = re.search(r'\b(20\d{2})[/-](2\d)\b', title)
    if m:
        return int(m.group(1)) + 1
    m = re.search(r'\b(20\d{2})[/-](20\d{2})\b', title)
    if m:
        return max(int(m.group(1)), int(m.group(2)))
    years = [int(y) for y in re.findall(r'\b(20[12]\d)\b', title)
             if 2015 <= int(y) <= 2030]
    return max(years) if years else None


def fiscal_year_from_filing(
    title: str, release_dt: str
) -> tuple[Optional[str], str]:
    """Determine fiscal year string from a filing's title and release datetime.

    Returns (fiscal_year_str, confidence) where confidence is:
      'HIGH'   — period-end date found in title
      'MEDIUM' — plain year in title, or Q1 heuristic applied
      'LOW'    — release year used as fallback

    Priority order:
    1. If title contains a period-end date (e.g. '31 August 2023'):
       - Sanity check: if release_date < period_end_date, the title year is
         one year ahead of the actual fiscal year (pre-period-end publication).
       - Otherwise use the period-end year directly.
    2. Title contains a plain year (e.g. 'Annual Report 2022') → use it.
    3. Generic title fallback: Q1 heuristic (UK reports published Jan–Apr
       almost always cover the prior fiscal year).
    """
    release_yr = release_dt[:4] if release_dt else ""
    if not release_yr:
        return None, "LOW"

    # Layer 1: period-end date in title
    period_end = extract_period_end_date(title)
    if period_end is not None:
        ped_year, ped_month, ped_day = period_end
        # Sanity check: was the filing published before the period end?
        try:
            release_date_str = release_dt[:10]  # YYYY-MM-DD
            release_parts = release_date_str.split("-")
            r_year, r_month, r_day = int(release_parts[0]), int(release_parts[1]), int(release_parts[2])
            pub_before_end = (
                (r_year, r_month, r_day) < (ped_year, ped_month, ped_day)
            )
        except (ValueError, IndexError):
            pub_before_end = False

        if pub_before_end:
            # Title year is the next FY; actual FY is one year earlier
            return str(ped_year - 1), "HIGH"
        else:
            return str(ped_year), "HIGH"

    # Layer 2: plain year in title
    fy = extract_fiscal_year(title)
    if fy is not None and fy <= int(release_yr) + 1:
        return str(fy), "MEDIUM"

    # Layer 3: Q1 heuristic for generic titles
    try:
        release_month = int(release_dt[5:7])
    except (ValueError, IndexError):
        release_month = 12
    if release_month <= 4:
        return str(int(release_yr) - 1), "MEDIUM"
    return release_yr, "LOW"


_TYPE_PRIORITY = {"10-K-ESEF": 2, "10-K": 1, "10-K-AFS": 0, "AR": -1}

# Patterns for detecting annual ER filings (Earnings Releases that are full-year results)
_ANNUAL_ER_RE = re.compile(
    r'final\s+results|year\s+ended|full[- ]year|annual\s+results|'
    r'annual\s+(report|financial)|preliminary\s+results|full\s+year\s+results|'
    r'results\s+for\s+the\s+(full\s+)?year|\bannual\b',
    re.IGNORECASE,
)
_INTERIM_ER_RE = re.compile(
    r'interim|half[\s-]year|half[\s-]yearly|h1\b|h2\b|'
    r'quarterly|quarter|trading\s+update|q[1-4]\b',
    re.IGNORECASE,
)


def is_annual_er(title: str) -> bool:
    """Return True if an ER-type filing looks like a full-year results announcement."""
    return bool(_ANNUAL_ER_RE.search(title)) and not bool(_INTERIM_ER_RE.search(title))


def select_winner(candidates: list[dict]) -> dict:
    """Prefer ESEF > 10-K > AFS > AR > annual-ER; break ties by highest pk."""
    def _rank(f: dict) -> tuple[int, int]:
        code = (f.get("filing_type") or {}).get("code", "")
        return (_TYPE_PRIORITY.get(code, -2), f["id"])
    return max(candidates, key=_rank)


def fetch_markdown(pk: str, name: str, fy: str, idx: int, total: int,
                   checkpoint: dict) -> str:
    """Fetch markdown for a pk. Updates checkpoint in-place. Returns new status."""
    cached_path = find_in_cache(pk)
    if cached_path:
        checkpoint[pk] = {"status": "cached", "size": cached_path.stat().st_size}
        print(f"  [{idx}/{total}] {pk} ({name} FY{fy}): cache hit")
        return "cached"

    try:
        resp = requests.get(
            f"{API_BASE}/filings/{pk}/markdown/",
            headers=HEADERS,
            timeout=60,
        )
        time.sleep(RATE_LIMIT_SEC)
    except requests.RequestException as exc:
        checkpoint[pk] = {"status": "request_error", "detail": str(exc)}
        print(f"  [{idx}/{total}] {pk} ({name} FY{fy}): request error: {exc}")
        return "request_error"

    if resp.status_code == 200:
        text = extract_markdown_text(resp)
        if text.strip():
            out_path = MD_DIR / f"{pk}.md"
            out_path.write_text(text, encoding="utf-8")
            size = len(text.encode())
            checkpoint[pk] = {"status": "fetched", "size": size}
            print(f"  [{idx}/{total}] {pk} ({name} FY{fy}): FETCHED ({size:,} bytes)")
            return "fetched"
        else:
            checkpoint[pk] = {"status": "empty"}
            print(f"  [{idx}/{total}] {pk} ({name} FY{fy}): empty")
            return "empty"

    elif resp.status_code == 404:
        proc_status, detail = extract_processing_status(resp)
        checkpoint[pk] = {"status": proc_status, "detail": detail}
        detail_str = f": {detail}" if detail else ""
        print(f"  [{idx}/{total}] {pk} ({name} FY{fy}): {proc_status}{detail_str}")
        return proc_status

    elif resp.status_code == 403:
        checkpoint[pk] = {"status": "access_denied"}
        print(f"  [{idx}/{total}] {pk} ({name} FY{fy}): 403 access denied")
        return "access_denied"

    else:
        status = f"http_{resp.status_code}"
        checkpoint[pk] = {"status": status}
        print(f"  [{idx}/{total}] {pk} ({name} FY{fy}): HTTP {resp.status_code}")
        return status


def rebuild_processing_status_csv(manifest_rows: list[dict], checkpoint: dict) -> None:
    rows = []
    for row in manifest_rows:
        pk = row["pk"]
        cp = checkpoint.get(pk, {})
        rows.append({
            "pk":                pk,
            "company__lei":      row["company__lei"],
            "company__name":     row["company__name"],
            "market_segment":    row["market_segment"],
            "fiscal_year":       row["fiscal_year"],
            "release_year":      row["release_year"],
            "filing_type__code": row["filing_type__code"],
            "is_esef":           row["is_esef"],
            "md_status":         cp.get("status", ""),
            "md_size":           cp.get("size", ""),
            "md_detail":         cp.get("detail", ""),
        })
    with open(PS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PS_FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["company__lei"], x["fiscal_year"])):
            w.writerow(r)


# ── Step A: refresh stale pks ─────────────────────────────────────────────────

def refresh_stale_pks(
    target_rows: list[dict],
    manifest_rows: list[dict],
    checkpoint: dict,
    dry_run: bool,
) -> tuple[int, int]:
    """Returns (cache_hits, newly_fetched)."""
    stale = [r for r in target_rows if r["fr_status"] in STALE_STATUSES and r["fr_pk"]]
    print(f"\n── Step A: Refresh stale pks ({len(stale)} rows) ─────────────────")
    before = Counter(r["fr_status"] for r in stale)
    for s, n in sorted(before.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n}")

    if not stale:
        print("  Nothing stale.")
        return 0, 0

    # Cache pass
    cache_hits = 0
    api_needed = []
    for row in stale:
        pk = row["fr_pk"]
        cached_path = find_in_cache(pk)
        if cached_path:
            checkpoint[pk] = {"status": "cached", "size": cached_path.stat().st_size}
            cache_hits += 1
        else:
            api_needed.append(row)

    print(f"\n  Cache hits: {cache_hits}  |  API calls needed: {len(api_needed)}")

    if dry_run:
        print("  Dry-run: skipping API calls.")
        return cache_hits, 0

    newly_fetched = 0
    status_counts = Counter()
    for i, row in enumerate(api_needed, start=1):
        result = fetch_markdown(
            row["fr_pk"], row["company_name"], row["fiscal_year"],
            i, len(api_needed), checkpoint,
        )
        status_counts[result] += 1
        if result in ("fetched", "cached"):
            newly_fetched += 1
        if i % 50 == 0:
            with open(P3_CHECKPOINT, "w") as f:
                json.dump(checkpoint, f)

    print(f"\n  API results: {dict(status_counts)}")
    return cache_hits, newly_fetched


# ── Step B: check not_in_fr companies ────────────────────────────────────────

def check_not_in_fr(
    target_rows: list[dict],
    manifest_rows: list[dict],
    checkpoint: dict,
    dry_run: bool,
) -> list[dict]:
    """Query FR for companies with no FR entry. Returns list of new manifest rows added."""
    not_in_fr_map: dict[str, str] = {}  # lei -> company_name
    for r in target_rows:
        if r["fr_status"] == "not_in_fr":
            not_in_fr_map[r["lei"]] = r["company_name"]

    print(f"\n── Step B: Check not_in_fr companies ({len(not_in_fr_map)} unique LEIs) ──")

    # Build set of existing pks to avoid duplicates
    existing_pks = {r["pk"] for r in manifest_rows}

    if dry_run:
        print("  Dry-run: skipping API calls.")
        return []

    new_manifest_rows: list[dict] = []
    found_count = 0

    leis = list(not_in_fr_map.items())
    for i, (lei, name) in enumerate(leis, start=1):
        try:
            r = api_get("/filings/", {
                "lei":      lei,
                "types":    "10-K,10-K-ESEF,10-K-AFS,AR,ER",
                "ordering": "-release_datetime",
                "page_size": 100,
            })
        except Exception as exc:
            print(f"  [{i}/{len(leis)}] {name}: network error — {exc}")
            continue

        if r.status_code != 200:
            if r.status_code not in (404, 200):
                print(f"  [{i}/{len(leis)}] {name}: HTTP {r.status_code}")
            continue

        filings = r.json().get("results", [])
        if not filings:
            continue

        # Group by fiscal year, same logic as phase 2
        by_fy: dict[str, list] = defaultdict(list)
        for f in filings:
            ftype = (f.get("filing_type") or {}).get("code", "")
            title = f.get("title", "")
            # For ER filings, only include annual results (not interim/quarterly)
            if ftype == "ER" and not is_annual_er(title):
                continue
            release_dt = f.get("release_datetime") or ""
            release_yr = release_dt[:4]
            if release_yr < MIN_YEAR:
                continue
            fiscal_yr, _conf = fiscal_year_from_filing(title, release_dt)
            if fiscal_yr is None or fiscal_yr not in TARGET_YEARS:
                continue
            f["_release_year"] = release_yr
            f["_fy_confidence"] = _conf
            by_fy[fiscal_yr].append(f)

        if not by_fy:
            continue

        year_entries = []
        for fy_key, candidates in sorted(by_fy.items()):
            winner = select_winner(candidates)
            pk_str = str(winner["id"])
            if pk_str in existing_pks:
                continue  # already in manifest
            ft = winner.get("filing_type") or {}
            release_yr = winner.get("_release_year", (winner.get("release_datetime") or "")[:4])
            entry = {
                "pk":                pk_str,
                "company__lei":      lei,
                "company__name":     name,
                "market_segment":    "",   # FR doesn't give us this reliably here
                "fiscal_year":       fy_key,
                "release_year":      release_yr,
                "release_datetime":  winner.get("release_datetime", ""),
                "title":             winner.get("title", ""),
                "filing_type__code": ft.get("code", ""),
                "filing_type__name": ft.get("name", ""),
                "is_esef":           ft.get("code") == "10-K-ESEF",
                "candidates_count":  len(candidates),
            }
            year_entries.append(entry)
            existing_pks.add(pk_str)

        if year_entries:
            found_count += 1
            yrs = [e["fiscal_year"] for e in year_entries]
            print(f"  [{i}/{len(leis)}] {name}: found {len(year_entries)} filing(s) — FY{', FY'.join(yrs)}")
            new_manifest_rows.extend(year_entries)

    print(f"\n  Companies newly found in FR: {found_count} / {len(not_in_fr_map)}")
    print(f"  New manifest rows: {len(new_manifest_rows)}")

    if not new_manifest_rows:
        return []

    # Append to manifest.csv
    with open(MANIFEST_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        for row in new_manifest_rows:
            w.writerow(row)
    print(f"  Appended {len(new_manifest_rows)} rows to manifest.csv")

    # Update phase2 checkpoint
    p2_cp: dict = {}
    if P2_CHECKPOINT.exists():
        with open(P2_CHECKPOINT) as f:
            p2_cp = json.load(f)
    for row in new_manifest_rows:
        lei = row["company__lei"]
        if lei not in p2_cp:
            p2_cp[lei] = {"status": "ok_refresh", "years": 0}
        p2_cp[lei]["years"] = p2_cp[lei].get("years", 0) + 1
    with open(P2_CHECKPOINT, "w") as f:
        json.dump(p2_cp, f)

    # Fetch markdown for new pks
    print(f"\n  Fetching markdown for {len(new_manifest_rows)} new filings...")
    for i, row in enumerate(new_manifest_rows, start=1):
        fetch_markdown(
            row["pk"], row["company__name"], row["fiscal_year"],
            i, len(new_manifest_rows), checkpoint,
        )
        if i % 20 == 0:
            with open(P3_CHECKPOINT, "w") as f:
                json.dump(checkpoint, f)

    return new_manifest_rows


# ── Step C: check year-gaps for partially-covered companies ──────────────────

def check_year_gaps(
    target_rows: list[dict],
    manifest_rows: list[dict],
    checkpoint: dict,
    dry_run: bool,
) -> tuple[list[dict], list[tuple]]:
    """Query FR for companies that have some FR entries but specific year-gaps.

    These are companies where FR has data for e.g. 2021, 2023, 2024 but our
    manifest shows not_in_fr for 2022 — likely missed during the original bulk
    fetch due to pagination or filings added to FR after our initial download.
    Returns (new_manifest_rows, mislabeled_pks).
    """
    # Build per-LEI status sets and gap years
    lei_statuses: dict[str, set] = defaultdict(set)
    lei_gap_years: dict[str, set] = defaultdict(set)
    lei_name: dict[str, str] = {}

    for r in target_rows:
        lei = r["lei"]
        lei_statuses[lei].add(r["fr_status"])
        lei_name[lei] = r["company_name"]
        if r["fr_status"] == "not_in_fr":
            lei_gap_years[lei].add(r["fiscal_year"])

    # Only companies with BOTH some FR rows AND some not_in_fr gaps
    gap_leis = {
        lei for lei, statuses in lei_statuses.items()
        if "not_in_fr" in statuses and (statuses - {"not_in_fr"})
    }

    total_gap_years = sum(len(lei_gap_years[lei]) for lei in gap_leis)
    print(f"\n── Step C: Check year-gaps ({len(gap_leis)} companies, "
          f"{total_gap_years} gap-years) ──")

    if not gap_leis:
        print("  No year-gaps found.")
        return [], []

    existing_pks = {r["pk"] for r in manifest_rows}
    pk_to_fy     = {r["pk"]: r["fiscal_year"] for r in manifest_rows}

    if dry_run:
        print("  Dry-run: skipping API calls.")
        for lei in sorted(gap_leis):
            print(f"    {lei_name[lei]}: missing {sorted(lei_gap_years[lei])}")
        return [], []

    new_manifest_rows: list[dict] = []
    mislabeled_pks: list[tuple] = []  # (pk, current_fy, correct_fy, lei, name)
    found_count = 0
    leis = sorted(gap_leis, key=lambda lei: lei_name[lei].lower())

    for i, lei in enumerate(leis, start=1):
        name = lei_name[lei]
        gap_years = lei_gap_years[lei]

        try:
            r = api_get("/filings/", {
                "lei":      lei,
                "types":    "10-K,10-K-ESEF,10-K-AFS,AR,ER",
                "ordering": "-release_datetime",
                "page_size": 100,
            })
        except Exception as exc:
            print(f"  [{i}/{len(leis)}] {name}: network error — {exc}")
            continue

        if r.status_code != 200:
            print(f"  [{i}/{len(leis)}] {name}: HTTP {r.status_code}")
            continue

        filings = r.json().get("results", [])
        if not filings:
            print(f"  [{i}/{len(leis)}] {name}: no filings returned (gap years: {sorted(gap_years)})")
            continue

        # Group all returned filings by fiscal year
        by_fy: dict[str, list] = defaultdict(list)
        for f in filings:
            ftype = (f.get("filing_type") or {}).get("code", "")
            title = f.get("title", "")
            # For ER filings, only include annual results (not interim/quarterly)
            if ftype == "ER" and not is_annual_er(title):
                continue
            release_dt = f.get("release_datetime") or ""
            release_yr = release_dt[:4]
            if release_yr < MIN_YEAR:
                continue
            fiscal_yr, fy_conf = fiscal_year_from_filing(title, release_dt)
            if fiscal_yr is None or fiscal_yr not in TARGET_YEARS:
                continue
            f["_release_year"] = release_yr
            f["_fy_confidence"] = fy_conf
            by_fy[fiscal_yr].append(f)

        # Check ALL FR-returned filings against ALL target years (not just gap_years).
        # This catches cascade mislabeling where a filing was assigned to the
        # wrong year and appears to fill that slot, leaving the correct year empty.
        # Only flag as mislabeled when confidence is HIGH or MEDIUM — LOW confidence
        # (release-year fallback) conflicts with markdown ground-truth corrections
        # for companies with non-December year-ends and non-Q1 publications.
        year_entries = []
        mislabeled: list[tuple[str, str, str]] = []  # (pk, current_year, correct_year)
        for fy_key in sorted(by_fy.keys()):
            if fy_key not in TARGET_YEARS:
                continue
            winner = select_winner(by_fy[fy_key])
            pk_str = str(winner["id"])
            if pk_str in existing_pks:
                # Filing already in manifest — check if it was assigned to correct year.
                # Only flag mislabeled when confidence is HIGH or MEDIUM: LOW confidence
                # means we used release-year fallback, which conflicts with markdown
                # ground-truth corrections for non-December, non-Q1 companies.
                current_fy = pk_to_fy.get(pk_str, "?")
                fy_conf = winner.get("_fy_confidence", "LOW")
                if current_fy != fy_key and fy_conf != "LOW":
                    mislabeled.append((pk_str, current_fy, fy_key))
                continue
            # Only add NEW rows for actual gap years
            if fy_key not in gap_years:
                continue
            ft = winner.get("filing_type") or {}
            release_yr = winner.get("_release_year", (winner.get("release_datetime") or "")[:4])
            entry = {
                "pk":                pk_str,
                "company__lei":      lei,
                "company__name":     name,
                "market_segment":    "",
                "fiscal_year":       fy_key,
                "release_year":      release_yr,
                "release_datetime":  winner.get("release_datetime", ""),
                "title":             winner.get("title", ""),
                "filing_type__code": ft.get("code", ""),
                "filing_type__name": ft.get("name", ""),
                "is_esef":           ft.get("code") == "10-K-ESEF",
                "candidates_count":  len(by_fy[fy_key]),
            }
            year_entries.append(entry)
            existing_pks.add(pk_str)

        fr_years = sorted(by_fy.keys())
        if year_entries or mislabeled:
            found_count += 1
            filled = [e["fiscal_year"] for e in year_entries]
            still_missing = sorted(gap_years - set(filled) - {m[2] for m in mislabeled})
            parts = []
            if filled:
                parts.append(f"filled FY{', FY'.join(filled)}")
            for pk_str, cur, correct in mislabeled:
                parts.append(f"MISLABELED pk={pk_str} (in manifest as FY{cur}, should be FY{correct})")
                mislabeled_pks.append((pk_str, cur, correct, lei, name))
            if still_missing:
                parts.append(f"still missing: {still_missing} (FR has: {fr_years})")
            print(f"  [{i}/{len(leis)}] {name}: {'; '.join(parts)}")
            new_manifest_rows.extend(year_entries)
        else:
            print(f"  [{i}/{len(leis)}] {name}: gap years {sorted(gap_years)} not in FR "
                  f"(FR has: {fr_years})")

    print(f"\n  Companies with gaps resolved: {found_count} / {len(leis)}")
    print(f"  New manifest rows:  {len(new_manifest_rows)}")
    print(f"  Mislabeled filings: {len(mislabeled_pks)}")
    if mislabeled_pks:
        print("  (run with --fix-labels to correct year assignments in manifest.csv)")

    if not new_manifest_rows and not mislabeled_pks:
        return [], []

    # Append to manifest.csv
    with open(MANIFEST_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        for row in new_manifest_rows:
            w.writerow(row)
    print(f"  Appended {len(new_manifest_rows)} rows to manifest.csv")

    # Update phase2 checkpoint
    p2_cp: dict = {}
    if P2_CHECKPOINT.exists():
        with open(P2_CHECKPOINT) as f:
            p2_cp = json.load(f)
    for row in new_manifest_rows:
        lei = row["company__lei"]
        if lei not in p2_cp:
            p2_cp[lei] = {"status": "ok_gap_fill", "years": 0}
        p2_cp[lei]["years"] = p2_cp[lei].get("years", 0) + 1
    with open(P2_CHECKPOINT, "w") as f:
        json.dump(p2_cp, f)

    # Fetch markdown for new pks
    print(f"\n  Fetching markdown for {len(new_manifest_rows)} new filings...")
    for i, row in enumerate(new_manifest_rows, start=1):
        fetch_markdown(
            row["pk"], row["company__name"], row["fiscal_year"],
            i, len(new_manifest_rows), checkpoint,
        )
        if i % 20 == 0:
            with open(P3_CHECKPOINT, "w") as f:
                json.dump(checkpoint, f)

    return new_manifest_rows, mislabeled_pks


def fix_mislabeled_years(mislabeled_pks: list[tuple]) -> int:
    """Correct fiscal_year for mislabeled rows in manifest.csv.

    mislabeled_pks entries: (pk, current_fy, correct_fy, lei, name)
    Returns number of rows corrected.
    """
    if not mislabeled_pks:
        return 0

    rows = []
    with open(MANIFEST_CSV) as f:
        rows = list(csv.DictReader(f))

    fix_map = {pk: correct for pk, _cur, correct, _lei, _name in mislabeled_pks}
    corrected = 0
    for row in rows:
        if row["pk"] in fix_map:
            old_fy = row["fiscal_year"]
            row["fiscal_year"] = fix_map[row["pk"]]
            print(f"  Correcting pk={row['pk']} ({row['company__name']}): "
                  f"FY{old_fy} → FY{row['fiscal_year']}")
            corrected += 1

    if corrected:
        fieldnames = rows[0].keys() if rows else MANIFEST_FIELDS
        with open(MANIFEST_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  Corrected {corrected} rows in manifest.csv")

    return corrected


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool, check_not_in_fr_flag: bool, check_year_gaps_flag: bool,
         fix_labels: bool, rebuild_manifest: bool) -> None:
    if not API_KEY and not dry_run:
        print("ERROR: FR_API_KEY not set in .env.local")
        sys.exit(1)

    MD_DIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    target_rows: list[dict] = []
    with open(TARGET_MANIFEST) as f:
        target_rows = list(csv.DictReader(f))

    checkpoint: dict = {}
    if P3_CHECKPOINT.exists():
        with open(P3_CHECKPOINT) as f:
            checkpoint = json.load(f)

    manifest_rows: list[dict] = []
    with open(MANIFEST_CSV) as f:
        manifest_rows = list(csv.DictReader(f))

    print(f"Target manifest: {len(target_rows):,} rows")
    print(f"FR manifest:     {len(manifest_rows):,} rows")
    print(f"P3 checkpoint:   {len(checkpoint):,} entries")

    # ── Step A ────────────────────────────────────────────────────────────────
    cache_hits, newly_fetched = refresh_stale_pks(
        target_rows, manifest_rows, checkpoint, dry_run
    )

    # ── Step B ────────────────────────────────────────────────────────────────
    new_rows_b: list[dict] = []
    if check_not_in_fr_flag:
        new_rows_b = check_not_in_fr(target_rows, manifest_rows, checkpoint, dry_run)
        manifest_rows.extend(new_rows_b)

    # ── Step C ────────────────────────────────────────────────────────────────
    new_rows_c: list[dict] = []
    all_mislabeled: list[tuple] = []
    if check_year_gaps_flag:
        # Reload target_rows so Step C sees any rows Step B may have resolved
        with open(TARGET_MANIFEST) as f:
            target_rows = list(csv.DictReader(f))
        new_rows_c, all_mislabeled = check_year_gaps(
            target_rows, manifest_rows, checkpoint, dry_run
        )
        manifest_rows.extend(new_rows_c)

    # ── Fix mislabeled years ──────────────────────────────────────────────────
    if fix_labels and all_mislabeled:
        print(f"\n── Fixing {len(all_mislabeled)} mislabeled year assignments ────────")
        fix_mislabeled_years(all_mislabeled)

    # ── Save checkpoint + rebuild CSVs ────────────────────────────────────────
    with open(P3_CHECKPOINT, "w") as f:
        json.dump(checkpoint, f)
    print(f"\nCheckpoint saved ({len(checkpoint)} entries)")

    rebuild_processing_status_csv(manifest_rows, checkpoint)
    print(f"Rebuilt {PS_CSV.relative_to(REPO_ROOT)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────────────")
    print(f"  Step A — stale pks:   cache hits={cache_hits}, newly fetched={newly_fetched}")
    if check_not_in_fr_flag:
        newly_found_md = sum(
            1 for r in new_rows_b
            if checkpoint.get(r["pk"], {}).get("status") in ("fetched", "cached")
        )
        print(f"  Step B — not_in_fr:   {len(new_rows_b)} new filings found, {newly_found_md} with MD")
    if check_year_gaps_flag:
        newly_found_md = sum(
            1 for r in new_rows_c
            if checkpoint.get(r["pk"], {}).get("status") in ("fetched", "cached")
        )
        print(f"  Step C — year-gaps:   {len(new_rows_c)} gap filings found, {newly_found_md} with MD")
        if all_mislabeled:
            fixed = len(all_mislabeled) if fix_labels else 0
            print(f"           mislabeled: {len(all_mislabeled)} detected, {fixed} corrected "
                  f"{'(use --fix-labels to correct)' if not fix_labels else ''}")

    if rebuild_manifest:
        print("\n── Rebuilding target_manifest.csv ────────────────────────────")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "build_target_manifest.py")],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            print("WARNING: build_target_manifest.py exited with non-zero code")
    else:
        print("\nNext: python scripts/build_target_manifest.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Cache check only, no API calls")
    p.add_argument("--check-not-in-fr", action="store_true",
                   help="Query FR API for companies completely absent from FR")
    p.add_argument("--check-year-gaps", action="store_true",
                   help="Query FR API for partial companies missing specific years")
    p.add_argument("--fix-labels", action="store_true",
                   help="Correct mislabeled fiscal years found during --check-year-gaps")
    p.add_argument("--rebuild-manifest", action="store_true",
                   help="Re-run build_target_manifest.py after updating")
    args = p.parse_args()
    main(
        dry_run=args.dry_run,
        check_not_in_fr_flag=args.check_not_in_fr,
        check_year_gaps_flag=args.check_year_gaps,
        fix_labels=args.fix_labels,
        rebuild_manifest=args.rebuild_manifest,
    )
