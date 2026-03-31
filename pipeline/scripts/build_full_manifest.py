#!/usr/bin/env python3
"""Build the definitive FULL_DB/manifest.csv.

For every company in FULL_DB/companies.csv this script determines, per
(lei, fiscal_year):

  1. ch_pdf_available  — is an annual-accounts PDF present on Companies House?
       * For ~1,401 LEIs already in ch_period_of_accounts.csv → 'yes' (trusted).
       * For the remaining ~54 LEIs the CH API is queried directly.

  2. local_markdown    — do we already hold a markdown file locally?
       Sources checked (highest-priority first):
         FR_consolidated/metadata.csv
         ch_gap_fill/gap_manifest.csv
         ch_gap_fill_2021/gap_manifest.csv
         ch_gap_fill_fy2020/gap_manifest.csv
         ch_gap_fill_2021_direct/gap_manifest.csv

  3. fr_available      — is this filing available in FR?
       Resolution order (cheapest first — stops as soon as a match is found):
         a) FR_dataset/manifest_raw.csv   — already-fetched FR coverage (~1,276 LEIs)
         b) FR live API search             — only for LEIs absent from manifest_raw.csv
                                             AND missing locally (typically ~100-150 LEIs)
       Live API calls use the correct parameters:
         GET /filings/?lei={lei}&types=10-K,10-K-ESEF,10-K-AFS,AR
       Results are cached in FULL_DB/fr_search_cache.jsonl (resumable).

Scope: filings with submission_date >= MIN_SUBMISSION_DATE (default 2020-01-01).

Outputs (written to data/FULL_DB/):
  manifest.csv          — one row per (lei, fiscal_year), full status
  fr_contact_report.csv — filings where ch_pdf_available=yes AND fr_api_available=no
                          (to send to FR asking for missing coverage)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
FULL_DB   = DATA_ROOT / "FULL_DB"

COMPANIES_CSV         = FULL_DB / "companies.csv"
CH_PERIOD_OF_ACCOUNTS = DATA_ROOT / "FR_dataset" / "ch_period_of_accounts.csv"
FR_MANIFEST_RAW       = DATA_ROOT / "FR_dataset" / "manifest_raw.csv"

FR_CONSOLIDATED_METADATA = DATA_ROOT / "FR_consolidated" / "metadata.csv"
GAP_MAIN_MANIFEST        = DATA_ROOT / "ch_gap_fill" / "gap_manifest.csv"
GAP_2021_MANIFEST        = DATA_ROOT / "ch_gap_fill_2021" / "gap_manifest.csv"
GAP_FY2020_MANIFEST      = DATA_ROOT / "ch_gap_fill_fy2020" / "gap_manifest.csv"
GAP_2021_DIRECT_MANIFEST = DATA_ROOT / "ch_gap_fill_2021_direct" / "gap_manifest.csv"

FR_SEARCH_CACHE = FULL_DB / "fr_search_cache.jsonl"
MANIFEST_CSV    = FULL_DB / "manifest.csv"
FR_CONTACT_CSV  = FULL_DB / "fr_contact_report.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FR_API_BASE            = "https://api.financialreports.eu"
FR_ANNUAL_TYPES_PARAM  = "10-K,10-K-ESEF,10-K-AFS,AR"
FR_ANNUAL_TYPE_CODES   = {"10-K", "10-K-ESEF", "10-K-AFS", "AR"}
FR_MATCH_WINDOW_DAYS   = 270  # release_datetime must be within [made_up_date, +270d]

CH_API_BASE       = "https://api.company-information.service.gov.uk"
CH_ACCOUNTS_TYPES = {"AA", "AA01", "AAMD"}
CH_ITEMS_PER_PAGE = 100

MIN_SUBMISSION_DATE = "2020-01-01"

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "lei",
    "company_name",
    "ch_company_number",
    "market_segment",
    "fiscal_year",
    "made_up_date",
    "submission_date",
    "ch_source",            # 'ch_poa_csv' | 'ch_api' | 'ch_api_error' | 'ch_api_not_found'
    "ch_pdf_available",     # 'yes' | 'no' | 'unknown'
    "local_markdown",       # 'yes' | 'no'
    "local_source",
    "local_path",
    "fr_source",            # 'manifest_raw' | 'live_api' | 'not_checked'
    "fr_api_available",     # 'yes' | 'no' | 'not_checked' | 'error'
    "fr_pk",
    "fr_type_code",
    "fr_release_datetime",
    "fr_title",
    "status",
]

FR_CONTACT_FIELDS = [
    "lei",
    "company_name",
    "ch_company_number",
    "market_segment",
    "fiscal_year",
    "made_up_date",
    "submission_date",
    "fr_pk",
    "fr_release_datetime",
]

STATUS_HAVE_MARKDOWN = "have_markdown"
STATUS_FR_AVAILABLE  = "fr_available"
STATUS_CH_ONLY       = "ch_only"        # CH confirmed; FR does not have it → contact FR
STATUS_MISSING_BOTH  = "missing_both"   # CH confirmed; no local; FR error
STATUS_NOT_ON_CH     = "not_on_ch"      # no CH accounts found
STATUS_UNKNOWN       = "unknown"        # CH API error

HAVE_MARKDOWN_STATUSES = {"fr_recovered", "ch_processed"}

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _parse_iso(raw: str) -> date | None:
    raw = str(raw or "").strip()[:10]
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _pair_key(lei: str, fiscal_year: str | int) -> tuple[str, str]:
    return (str(lei).strip(), str(fiscal_year).strip())


def _resolve_path(raw: str) -> Path | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    if raw.startswith("../data/"):
        return REPO_ROOT / raw[3:]
    if raw.startswith("data/"):
        return REPO_ROOT / raw
    return REPO_ROOT / raw


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class RateLimiter:
    def __init__(self, rps: float) -> None:
        self.interval = 0.0 if rps <= 0 else 1.0 / rps
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                sleep_for = self._next - now
                self._next += self.interval
            else:
                sleep_for = 0.0
                self._next = now + self.interval
        if sleep_for > 0:
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Step 1 — Master company list
# ---------------------------------------------------------------------------

def load_companies() -> dict[str, dict]:
    result: dict[str, dict] = {}
    with COMPANIES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lei = str(row.get("lei") or "").strip()
            if not lei:
                continue
            ch = str(row.get("ch_company_number") or "").strip()
            if ch.isdigit():
                ch = ch.zfill(8)
            result[lei] = {
                "lei": lei,
                "company_name": str(row.get("company_name") or "").strip(),
                "ch_company_number": ch,
                "market_segment": str(row.get("market_segment") or "").strip(),
            }
    return result


# ---------------------------------------------------------------------------
# Step 2 — CH period-of-accounts CSV
# ---------------------------------------------------------------------------

def load_ch_poa_universe(min_sub: str = MIN_SUBMISSION_DATE) -> dict[tuple[str, str], dict]:
    """Load expected (lei, fiscal_year) rows from ch_period_of_accounts.csv.

    Filter: submission_date >= min_sub.
    Dedup: prefer AA; keep latest submission_date.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    with CH_PERIOD_OF_ACCOUNTS.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            lei = str(raw.get("lei") or "").strip()
            fy  = str(raw.get("fiscal_year") or "").strip()
            sub = str(raw.get("submission_date") or "").strip()
            if not lei or not fy or sub < min_sub:
                continue
            ch = str(raw.get("ch_company_number") or "").strip()
            if ch.isdigit():
                ch = ch.zfill(8)
            groups[_pair_key(lei, fy)].append({
                "lei": lei,
                "company_name": str(raw.get("name") or raw.get("company_name") or "").strip(),
                "ch_company_number": ch,
                "fiscal_year": fy,
                "made_up_date": str(raw.get("made_up_date") or "").strip(),
                "submission_date": sub,
                "_filing_type": str(raw.get("filing_type") or "").strip(),
            })

    result: dict[tuple[str, str], dict] = {}
    for key, group in groups.items():
        aa   = [r for r in group if r["_filing_type"] == "AA"]
        pool = aa if aa else group
        best = dict(max(pool, key=lambda r: r["submission_date"]))
        best.pop("_filing_type", None)
        best["_ch_source"] = "ch_poa_csv"
        result[key] = best

    return result


# ---------------------------------------------------------------------------
# Step 3 — CH API for companies absent from ch_period_of_accounts.csv
# ---------------------------------------------------------------------------

def _ch_headers(api_key: str) -> dict:
    from base64 import b64encode
    creds = b64encode(f"{api_key}:".encode()).decode()
    return {"Authorization": f"Basic {creds}", "Accept": "application/json"}


def _fetch_ch_history(ch_number: str, headers: dict, limiter: RateLimiter, timeout: int = 30) -> list[dict]:
    """Return all accounts filings for a CH company number (handles pagination)."""
    items: list[dict] = []
    start = 0
    while True:
        limiter.wait()
        url    = f"{CH_API_BASE}/company/{ch_number}/filing-history"
        params = {"category": "accounts", "items_per_page": CH_ITEMS_PER_PAGE, "start_index": start}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(str(exc)) from exc
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        body  = resp.json()
        batch = body.get("items") or []
        items.extend(batch)
        total = int(body.get("total_count") or 0)
        start += len(batch)
        if start >= total or not batch:
            break
    return items


def _ch_items_to_rows(lei: str, company_name: str, ch_number: str, items: list[dict], min_sub: str) -> list[dict]:
    best_by_fy: dict[str, dict] = {}
    for item in items:
        ftype    = str(item.get("type") or "").strip().upper()
        sub_date = str(item.get("date") or "").strip()
        if ftype not in CH_ACCOUNTS_TYPES or sub_date < min_sub:
            continue
        desc_vals  = item.get("description_values") or {}
        made_up_dt = _parse_iso(str(desc_vals.get("made_up_date") or ""))
        if made_up_dt is None:
            sub_dt = _parse_iso(sub_date)
            if sub_dt is None:
                continue
            made_up_dt = date(sub_dt.year - 1, 12, 31)
        fy  = str(made_up_dt.year)
        row = {
            "lei": lei, "company_name": company_name, "ch_company_number": ch_number,
            "fiscal_year": fy, "made_up_date": made_up_dt.isoformat(),
            "submission_date": sub_date, "_filing_type": ftype, "_ch_source": "ch_api",
        }
        existing = best_by_fy.get(fy)
        if existing is None:
            best_by_fy[fy] = row
        elif ftype == "AA" and existing["_filing_type"] != "AA":
            best_by_fy[fy] = row
        elif sub_date > existing["submission_date"]:
            best_by_fy[fy] = row

    rows = []
    for row in best_by_fy.values():
        r = dict(row)
        r.pop("_filing_type", None)
        rows.append(r)
    return rows


def fetch_ch_api_universe(
    missing_leis: list[str],
    companies: dict[str, dict],
    api_key: str,
    *,
    rps: float = 1.5,
    min_sub: str = MIN_SUBMISSION_DATE,
) -> tuple[dict[tuple[str, str], dict], dict[str, str], set[str]]:
    headers = _ch_headers(api_key)
    limiter = RateLimiter(rps)
    new_rows: dict[tuple[str, str], dict] = {}
    errors:   dict[str, str] = {}
    no_filings: set[str] = set()

    print(f"\nQuerying CH API for {len(missing_leis)} companies not in ch_period_of_accounts.csv …")
    for lei in tqdm(missing_leis, desc="CH API", unit="co"):
        meta         = companies.get(lei, {})
        ch_number    = meta.get("ch_company_number", "").strip()
        company_name = meta.get("company_name", lei)
        if not ch_number:
            errors[lei] = "no_ch_number"
            continue
        try:
            items = _fetch_ch_history(ch_number, headers, limiter)
        except Exception as exc:
            errors[lei] = str(exc)[:200]
            continue
        rows = _ch_items_to_rows(lei, company_name, ch_number, items, min_sub)
        if not rows:
            no_filings.add(lei)
        for row in rows:
            key = _pair_key(row["lei"], row["fiscal_year"])
            if key not in new_rows:
                new_rows[key] = row

    return new_rows, errors, no_filings


# ---------------------------------------------------------------------------
# Step 4 — Local markdown corpus
# ---------------------------------------------------------------------------

def load_local_corpus() -> dict[tuple[str, str], dict]:
    pairs: dict[tuple[str, str], dict] = {}

    if FR_CONSOLIDATED_METADATA.exists():
        with FR_CONSOLIDATED_METADATA.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lei  = str(row.get("lei") or "").strip()
                fy   = str(row.get("fiscal_year") or "").strip()
                path = _resolve_path(row.get("src_path", ""))
                if not lei or not fy or not path or not path.exists():
                    continue
                pairs[_pair_key(lei, fy)] = {
                    "local_markdown": "yes", "local_source": "fr_consolidated",
                    "local_path": str(path), "local_fr_pk": str(row.get("pk") or "").strip(),
                }

    for manifest_path, label in (
        (GAP_MAIN_MANIFEST,      "ch_gap_fill"),
        (GAP_2021_MANIFEST,      "ch_gap_fill_2021"),
        (GAP_FY2020_MANIFEST,    "ch_gap_fill_fy2020"),
    ):
        if not manifest_path.exists():
            continue
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") not in HAVE_MARKDOWN_STATUSES:
                    continue
                lei  = str(row.get("lei") or "").strip()
                fy   = str(row.get("fiscal_year") or "").strip()
                path = _resolve_path(row.get("markdown_path", ""))
                if not lei or not fy or not path or not path.exists():
                    continue
                key = _pair_key(lei, fy)
                if key in pairs:
                    continue
                pairs[key] = {
                    "local_markdown": "yes", "local_source": label,
                    "local_path": str(path), "local_fr_pk": str(row.get("pk") or "").strip(),
                }

    if GAP_2021_DIRECT_MANIFEST.exists():
        with GAP_2021_DIRECT_MANIFEST.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ch_processed":
                    continue
                lei  = str(row.get("lei") or "").strip()
                fy   = str(row.get("fiscal_year") or "").strip()
                path = _resolve_path(row.get("markdown_path", ""))
                if not lei or not fy or not path or not path.exists():
                    continue
                key = _pair_key(lei, fy)
                if key in pairs:
                    continue
                pairs[key] = {
                    "local_markdown": "yes", "local_source": "ch_gap_fill_2021_direct",
                    "local_path": str(path), "local_fr_pk": "",
                }

    return pairs


# ---------------------------------------------------------------------------
# Step 5a — FR coverage from manifest_raw.csv (no API calls)
# ---------------------------------------------------------------------------

def load_fr_manifest_coverage(
    min_sub: str = MIN_SUBMISSION_DATE,
) -> dict[tuple[str, str], dict]:
    """Build FR coverage dict from the already-fetched manifest_raw.csv.

    Returns {(lei, fiscal_year): {fr_pk, fr_type_code, fr_release_datetime, fr_title}}
    for every annual report filing in manifest_raw.csv whose release_datetime >= min_sub.

    For duplicate (lei, fiscal_year) keeps the highest-priority type
    (10-K-ESEF > 10-K > 10-K-AFS > AR) then highest pk.
    """
    TYPE_PRIORITY = {"10-K-ESEF": 4, "10-K": 3, "10-K-AFS": 2, "AR": 1}

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with FR_MANIFEST_RAW.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lei       = str(row.get("company__lei") or "").strip()
            fy        = str(row.get("fiscal_year") or "").strip()
            ftype     = str(row.get("filing_type__code") or "").strip()
            rel_dt    = str(row.get("release_datetime") or "").strip()
            if not lei or not fy or ftype not in FR_ANNUAL_TYPE_CODES:
                continue
            if rel_dt[:10] < min_sub:
                continue
            groups[_pair_key(lei, fy)].append({
                "fr_pk":               str(row.get("pk") or "").strip(),
                "fr_type_code":        ftype,
                "fr_release_datetime": rel_dt,
                "fr_title":            str(row.get("title") or "").strip(),
            })

    result: dict[tuple[str, str], dict] = {}
    for key, candidates in groups.items():
        winner = max(
            candidates,
            key=lambda c: (TYPE_PRIORITY.get(c["fr_type_code"], 0), int(c["fr_pk"] or 0)),
        )
        result[key] = winner
    return result


# ---------------------------------------------------------------------------
# Step 5b — FR live API for LEIs absent from manifest_raw.csv
# ---------------------------------------------------------------------------

def load_fr_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not FR_SEARCH_CACHE.exists():
        return cache
    with FR_SEARCH_CACHE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            lei = str(record.get("lei") or "").strip()
            if lei:
                cache[lei] = record
    return cache


def _append_fr_cache(record: dict) -> None:
    FULL_DB.mkdir(parents=True, exist_ok=True)
    with FR_SEARCH_CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fetch_fr_filings_for_lei(lei: str, api_key: str, limiter: RateLimiter, timeout: int = 60) -> dict:
    """Fetch all annual-report filings for a LEI from the FR live API.

    Uses the correct parameters: lei= and types= (not company__lei / filing_type__code).
    The date field in responses is release_datetime (not filing_date).
    """
    headers = {"x-api-key": api_key, "Accept": "application/json"}
    params  = {
        "lei":      lei,
        "types":    FR_ANNUAL_TYPES_PARAM,
        "ordering": "-release_datetime",
        "page_size": 100,
    }
    url     = f"{FR_API_BASE}/filings/"
    filings: list[dict] = []
    req_count = 0

    while url:
        limiter.wait()
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        except requests.RequestException as exc:
            return {"lei": lei, "error": f"request_error:{exc}", "filings": [], "request_count": req_count}
        req_count += 1
        if resp.status_code == 429:
            return {"lei": lei, "error": "http_429", "filings": [], "request_count": req_count}
        if resp.status_code == 404:
            return {"lei": lei, "error": "", "filings": [], "request_count": req_count}
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            return {"lei": lei, "error": f"http_{resp.status_code}", "filings": [], "request_count": req_count}

        ct   = (resp.headers.get("Content-Type") or "").lower()
        body = resp.json() if "json" in ct else {}
        for f in (body.get("results") or []):
            fid   = str(f.get("id") or "").strip()
            ftype = (f.get("filing_type") or {}).get("code", "") if isinstance(f.get("filing_type"), dict) else str(f.get("filing_type__code") or "")
            if fid and ftype in FR_ANNUAL_TYPE_CODES:
                filings.append({
                    "pk":                  fid,
                    "fr_type_code":        ftype,
                    "fr_release_datetime": str(f.get("release_datetime") or "").strip(),
                    "fr_title":            str(f.get("title") or "").strip(),
                })
        next_url = (body.get("next") or "") if isinstance(body, dict) else ""
        url = str(next_url).strip()
        if url.startswith("/"):
            url = f"{FR_API_BASE}{url}"
        params = None

    return {"lei": lei, "error": "", "filings": filings, "request_count": req_count}


def run_fr_live_searches(
    leis: list[str],
    cache: dict[str, dict],
    api_key: str,
    *,
    workers: int = 8,
    rps: float = 10.0,
) -> dict[str, dict]:
    """Fetch FR filings for LEIs not already in the live cache."""
    to_fetch = [lei for lei in leis if lei not in cache]
    if not to_fetch:
        print("  FR live cache complete — no new API searches needed.")
        return cache

    print(f"\nFR live API: searching {len(to_fetch)} LEIs ({len(cache)} already cached) …")
    limiter = RateLimiter(rps)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_fr_filings_for_lei, lei, api_key, limiter): lei for lei in to_fetch}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="FR live", unit="lei"):
            record = fut.result()
            _append_fr_cache(record)
            cache[record["lei"]] = record

    return cache


# ---------------------------------------------------------------------------
# Step 5c — Match live FR results to CH rows by date window
# ---------------------------------------------------------------------------

def _match_live_fr(
    rows: list[dict],
    filings: list[dict],
    window: int = FR_MATCH_WINDOW_DAYS,
) -> dict[tuple[str, str], dict]:
    """Match CH rows to live FR filings by proximity of release_datetime to made_up_date."""
    TYPE_PRIORITY = {"10-K-ESEF": 4, "10-K": 3, "10-K-AFS": 2, "AR": 1}
    matches: dict[tuple[str, str], dict] = {}
    used: set[str] = set()

    for row in sorted(rows, key=lambda r: (r.get("made_up_date", ""), r["fiscal_year"])):
        made_up = _parse_iso(row.get("made_up_date", ""))
        if made_up is None:
            continue

        candidates: list[tuple[tuple, dict]] = []
        for f in filings:
            fid = f["pk"]
            if fid in used:
                continue
            rel_dt = _parse_iso(f.get("fr_release_datetime", ""))
            if rel_dt is None:
                continue
            delta = (rel_dt - made_up).days
            if not (0 <= delta <= window):
                continue
            type_rank = -TYPE_PRIORITY.get(f.get("fr_type_code", ""), 0)  # lower = better
            score = (type_rank, delta, fid)
            candidates.append((score, f))

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        winner = candidates[0][1]
        used.add(winner["pk"])
        matches[_pair_key(row["lei"], row["fiscal_year"])] = winner

    return matches


# ---------------------------------------------------------------------------
# Step 6 — Assemble manifest
# ---------------------------------------------------------------------------

def assemble_manifest(
    companies:        dict[str, dict],
    expected:         dict[tuple[str, str], dict],
    ch_api_errors:    dict[str, str],
    no_ch_filings:    set[str],
    local_corpus:     dict[tuple[str, str], dict],
    fr_manifest_cov:  dict[tuple[str, str], dict],   # from manifest_raw.csv
    fr_live_cache:    dict[str, dict],               # from live API
) -> list[dict]:

    # Pre-compute live FR matches for non-local pairs not covered by manifest_raw
    missing_by_lei: dict[str, list[dict]] = defaultdict(list)
    manifest_leis = {lei for lei, _ in fr_manifest_cov}
    for key, ch_row in expected.items():
        lei = ch_row["lei"]
        if key not in local_corpus and lei not in manifest_leis and key not in fr_manifest_cov:
            missing_by_lei[lei].append(ch_row)

    live_matches: dict[tuple[str, str], dict] = {}
    for lei, lei_rows in missing_by_lei.items():
        record  = fr_live_cache.get(lei, {})
        filings = record.get("filings", []) if not record.get("error") else []
        live_matches.update(_match_live_fr(lei_rows, filings))

    rows: list[dict] = []

    for key in sorted(expected.keys(), key=lambda k: (k[1], k[0])):
        ch_row  = expected[key]
        lei     = ch_row["lei"]
        company = companies.get(lei, {})
        local   = local_corpus.get(key)

        if local:
            rows.append({
                **_base(ch_row, company),
                "local_markdown": "yes",
                "local_source":   local["local_source"],
                "local_path":     local["local_path"],
                "fr_source":      "not_checked",
                "fr_api_available": "not_checked",
                "fr_pk":          local.get("local_fr_pk", ""),
                "fr_type_code":   "",
                "fr_release_datetime": "",
                "fr_title":       "",
                "status":         STATUS_HAVE_MARKDOWN,
            })
            continue

        # Try manifest_raw first
        fr_row = fr_manifest_cov.get(key)
        if fr_row:
            rows.append({
                **_base(ch_row, company),
                "local_markdown": "no",
                "local_source":   "",
                "local_path":     "",
                "fr_source":      "manifest_raw",
                "fr_api_available": "yes",
                "fr_pk":          fr_row["fr_pk"],
                "fr_type_code":   fr_row["fr_type_code"],
                "fr_release_datetime": fr_row["fr_release_datetime"],
                "fr_title":       fr_row["fr_title"],
                "status":         STATUS_FR_AVAILABLE,
            })
            continue

        # Try live API result
        fr_rec = fr_live_cache.get(lei, {})
        fr_err = fr_rec.get("error", "") if fr_rec else ""
        live   = live_matches.get(key)

        if fr_err:
            rows.append({
                **_base(ch_row, company),
                "local_markdown": "no", "local_source": "", "local_path": "",
                "fr_source": "live_api", "fr_api_available": "error",
                "fr_pk": "", "fr_type_code": "", "fr_release_datetime": "", "fr_title": "",
                "status": STATUS_MISSING_BOTH,
            })
        elif live:
            rows.append({
                **_base(ch_row, company),
                "local_markdown": "no", "local_source": "", "local_path": "",
                "fr_source": "live_api", "fr_api_available": "yes",
                "fr_pk": live["pk"], "fr_type_code": live.get("fr_type_code", ""),
                "fr_release_datetime": live.get("fr_release_datetime", ""),
                "fr_title": live.get("fr_title", ""),
                "status": STATUS_FR_AVAILABLE,
            })
        elif lei in manifest_leis or fr_rec:
            # LEI was searched (either in manifest_raw or live) and this (lei,fy) not found
            rows.append({
                **_base(ch_row, company),
                "local_markdown": "no", "local_source": "", "local_path": "",
                "fr_source": "manifest_raw" if lei in manifest_leis else "live_api",
                "fr_api_available": "no",
                "fr_pk": "", "fr_type_code": "", "fr_release_datetime": "", "fr_title": "",
                "status": STATUS_CH_ONLY,
            })
        else:
            # LEI not in manifest_raw and no live search done (shouldn't happen in full run)
            rows.append({
                **_base(ch_row, company),
                "local_markdown": "no", "local_source": "", "local_path": "",
                "fr_source": "not_checked", "fr_api_available": "not_checked",
                "fr_pk": "", "fr_type_code": "", "fr_release_datetime": "", "fr_title": "",
                "status": STATUS_UNKNOWN,
            })

    # CH API errors
    for lei, _error in ch_api_errors.items():
        company = companies.get(lei, {})
        rows.append({
            "lei": lei, "company_name": company.get("company_name", lei),
            "ch_company_number": company.get("ch_company_number", ""),
            "market_segment": company.get("market_segment", ""),
            "fiscal_year": "", "made_up_date": "", "submission_date": "",
            "ch_source": "ch_api_error", "ch_pdf_available": "unknown",
            "local_markdown": "no", "local_source": "", "local_path": "",
            "fr_source": "not_checked", "fr_api_available": "not_checked",
            "fr_pk": "", "fr_type_code": "", "fr_release_datetime": "", "fr_title": "",
            "status": STATUS_UNKNOWN,
        })

    # Confirmed no CH filings (offshore etc.)
    leis_covered = {r["lei"] for r in rows}
    for lei in no_ch_filings:
        if lei in leis_covered:
            continue
        company = companies.get(lei, {})
        rows.append({
            "lei": lei, "company_name": company.get("company_name", lei),
            "ch_company_number": company.get("ch_company_number", ""),
            "market_segment": company.get("market_segment", ""),
            "fiscal_year": "", "made_up_date": "", "submission_date": "",
            "ch_source": "ch_api", "ch_pdf_available": "no",
            "local_markdown": "no", "local_source": "", "local_path": "",
            "fr_source": "not_checked", "fr_api_available": "not_checked",
            "fr_pk": "", "fr_type_code": "", "fr_release_datetime": "", "fr_title": "",
            "status": STATUS_NOT_ON_CH,
        })

    return rows


def _base(ch_row: dict, company: dict) -> dict:
    return {
        "lei":               ch_row["lei"],
        "company_name":      ch_row.get("company_name") or company.get("company_name", ""),
        "ch_company_number": ch_row.get("ch_company_number") or company.get("ch_company_number", ""),
        "market_segment":    company.get("market_segment", ""),
        "fiscal_year":       ch_row["fiscal_year"],
        "made_up_date":      ch_row.get("made_up_date", ""),
        "submission_date":   ch_row.get("submission_date", ""),
        "ch_source":         ch_row.get("_ch_source", "ch_poa_csv"),
        "ch_pdf_available":  "yes",
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict], total_companies: int) -> None:
    filing_rows  = [r for r in rows if r["fiscal_year"]]
    status_cnt   = Counter(r["status"] for r in rows)
    fr_src_cnt   = Counter(r["fr_source"] for r in rows if r["fr_api_available"] == "yes")
    contact_count = status_cnt.get(STATUS_CH_ONLY, 0)

    print("\n" + "=" * 62)
    print("MANIFEST SUMMARY")
    print("=" * 62)
    print(f"  Total companies in scope          : {total_companies}")
    print(f"  (lei, fiscal_year) pairs expected : {len(filing_rows)}")
    print()
    print("  Status breakdown:")
    for s, n in sorted(status_cnt.items(), key=lambda x: -x[1]):
        print(f"    {s:<22} {n:>5}")
    if fr_src_cnt:
        print()
        print("  FR coverage source (for fr_available rows):")
        for s, n in sorted(fr_src_cnt.items(), key=lambda x: -x[1]):
            print(f"    {s:<20} {n:>5}")
    print("=" * 62)
    print(f"\nOutputs → {FULL_DB}/")
    print(f"  manifest.csv          ({len(filing_rows)} filing rows + {len(rows) - len(filing_rows)} company-level rows)")
    print(f"  fr_contact_report.csv ({contact_count} filings: CH has PDF, FR does not)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-submission-date", default=MIN_SUBMISSION_DATE,
                   help="Earliest submission_date to include (default: %(default)s)")
    p.add_argument("--fr-workers", type=int, default=8,
                   help="Parallel threads for FR live API searches (default: %(default)s)")
    p.add_argument("--fr-rps", type=float, default=10.0,
                   help="FR live API requests per second (default: %(default)s)")
    p.add_argument("--ch-rps", type=float, default=1.5,
                   help="CH API requests per second (default: %(default)s)")
    p.add_argument("--overwrite-fr-cache", action="store_true",
                   help="Delete and rebuild the live FR search cache from scratch")
    p.add_argument("--skip-ch-api", action="store_true",
                   help="Do not call CH API for the ~54 missing companies")
    p.add_argument("--skip-fr-live", action="store_true",
                   help="Do not call FR live API (uses manifest_raw.csv and local cache only)")
    p.add_argument("--dry-run", action="store_true",
                   help="Load all data and print summary, but skip ALL API calls and file writes")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(REPO_ROOT / ".env.local")
    fr_api_key = os.environ.get("FR_API_KEY", "").strip()
    ch_api_key = os.environ.get("COMPANIES_HOUSE_API_KEY", "").strip()

    if not fr_api_key and not args.dry_run and not args.skip_fr_live:
        sys.exit("ERROR: FR_API_KEY not set in .env.local")
    if not ch_api_key and not args.skip_ch_api and not args.dry_run:
        sys.exit("ERROR: COMPANIES_HOUSE_API_KEY not set in .env.local (use --skip-ch-api to bypass)")

    # 1 — master list
    print("Loading master company list …")
    companies = load_companies()
    print(f"  {len(companies)} companies")

    # 2 — CH POA CSV
    print(f"\nLoading ch_period_of_accounts.csv (submission_date >= {args.min_submission_date}) …")
    poa = load_ch_poa_universe(args.min_submission_date)
    poa_leis = {lei for lei, _ in poa}
    print(f"  {len(poa):,} (lei, fiscal_year) pairs | {len(poa_leis):,} unique LEIs")

    # 3 — CH API for missing LEIs
    missing_leis = sorted(set(companies) - poa_leis)
    print(f"\n  {len(missing_leis)} LEIs not in ch_period_of_accounts.csv")

    ch_new_rows: dict[tuple[str, str], dict] = {}
    ch_errors:   dict[str, str] = {}
    no_ch:       set[str] = set()

    if missing_leis and not args.skip_ch_api and not args.dry_run and ch_api_key:
        ch_new_rows, ch_errors, no_ch = fetch_ch_api_universe(
            missing_leis, companies, ch_api_key, rps=args.ch_rps, min_sub=args.min_submission_date,
        )
        print(f"  CH API: {len(ch_new_rows)} new (lei,fy) pairs | {len(no_ch)} no-CH-filings | {len(ch_errors)} errors")
    elif args.skip_ch_api or args.dry_run:
        print(f"  (CH API skipped)")
        for lei in missing_leis:
            ch_errors[lei] = "skipped"

    expected: dict[tuple[str, str], dict] = {**poa, **ch_new_rows}
    print(f"\nTotal expected (lei, fiscal_year) pairs: {len(expected):,}")

    # 4 — local corpus
    print("\nLoading local markdown corpus …")
    local_corpus = load_local_corpus()
    print(f"  {len(local_corpus):,} pairs already held locally")

    # 5a — FR manifest_raw.csv coverage
    print("\nLoading FR coverage from manifest_raw.csv …")
    fr_manifest_cov = load_fr_manifest_coverage(args.min_submission_date)
    fr_manifest_leis = {lei for lei, _ in fr_manifest_cov}
    print(f"  {len(fr_manifest_cov):,} (lei, fiscal_year) pairs in manifest_raw | {len(fr_manifest_leis):,} unique LEIs")

    # 5b — FR live API for LEIs not in manifest_raw AND with missing pairs
    missing_not_local = {k for k in expected if k not in local_corpus}
    leis_needing_live = sorted(
        {expected[k]["lei"] for k in missing_not_local}
        - fr_manifest_leis
    )
    print(f"\n  {len(missing_not_local):,} pairs missing locally")
    print(f"  {len(fr_manifest_leis):,} LEIs covered by manifest_raw → {len(leis_needing_live):,} LEIs need FR live search")

    if args.overwrite_fr_cache and FR_SEARCH_CACHE.exists():
        FR_SEARCH_CACHE.unlink()
        print("  FR live cache cleared (--overwrite-fr-cache).")

    fr_live_cache = load_fr_cache()
    print(f"  {len(fr_live_cache):,} LEIs already in live cache.")

    if not args.skip_fr_live and not args.dry_run and leis_needing_live:
        fr_live_cache = run_fr_live_searches(
            leis_needing_live, fr_live_cache, fr_api_key,
            workers=args.fr_workers, rps=args.fr_rps,
        )
    elif args.dry_run or args.skip_fr_live:
        print(f"  (FR live search skipped — {len(leis_needing_live)} LEIs would need searching)")

    # 6 — assemble
    print("\nAssembling manifest …")
    manifest_rows = assemble_manifest(
        companies, expected, ch_errors, no_ch, local_corpus, fr_manifest_cov, fr_live_cache,
    )

    contact_rows = [
        {f: r.get(f, "") for f in FR_CONTACT_FIELDS}
        for r in manifest_rows
        if r["status"] == STATUS_CH_ONLY
    ]

    print_summary(manifest_rows, len(companies))

    if not args.dry_run:
        _write_csv(MANIFEST_CSV, MANIFEST_FIELDS, manifest_rows)
        _write_csv(FR_CONTACT_CSV, FR_CONTACT_FIELDS, contact_rows)
        print(f"\nWrote: {MANIFEST_CSV}")
        print(f"Wrote: {FR_CONTACT_CSV}")
    else:
        print("\n(dry-run — no API calls made, no files written)")


if __name__ == "__main__":
    main()
