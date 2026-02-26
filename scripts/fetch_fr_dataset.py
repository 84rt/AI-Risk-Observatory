#!/usr/bin/env python3
"""Fetch the complete FR dataset for all LSE-listed companies.

Phases
------
1. Company list
   GET /companies/?listed_stock_exchange=10 (all pages)
   → data/FR_dataset/companies.json   (full API objects)
   → data/FR_dataset/companies.csv    (key fields + market_segment)

2. Filing manifest
   For each company: GET /filings/?lei=...&types=10-K,10-K-ESEF
   Pick one filing per company × release_year (ESEF preferred, highest id as tie-break)
   → data/FR_dataset/manifest.csv
   → data/FR_dataset/phase2_checkpoint.json  (resumable)

3. Markdown collection
   For each manifest entry:
     - Check cache dirs (FR_clean, FR-UK-2021-2023-test-2, FR_2026-02-05)
     - If not cached: GET /filings/{pk}/markdown/
       200 → save to data/FR_dataset/markdown/{pk}.md
       404 → record processing_status from response body
   → data/FR_dataset/processing_status.csv
   → data/FR_dataset/phase3_checkpoint.json  (resumable)

4. Summary
   → data/FR_dataset/summary.json

Usage
-----
    cd "AI Risk Observatory"
    source pipeline/venv/bin/activate
    python scripts/fetch_fr_dataset.py

Flags
-----
    --phase 1|2|3      Run only a specific phase (default: all)
    --dry-run          Phase 3 only: check cache coverage, skip API calls
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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

HEADERS = {
    "x-api-key": API_KEY,
    "Accept": "application/json",
}

OUT_DIR = REPO_ROOT / "data" / "FR_dataset"

# Cache directories searched in order before making an API call
CACHE_DIRS = [
    REPO_ROOT / "data" / "FR_clean"                  / "markdown",
    REPO_ROOT / "data" / "FR-UK-2021-2023-test-2"    / "markdown",
    REPO_ROOT / "data" / "FR_2026-02-05"              / "markdown",
]

LSE_EXCHANGE_ID = 10        # London Stock Exchange id in FR API
MIN_YEAR        = "2021"    # only include filings released 2021-present
RATE_LIMIT_SEC  = 0.4       # pause between requests

# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, params: Optional[dict] = None,
            retries: int = 3, backoff: float = 5.0) -> requests.Response:
    """GET with automatic retry on timeout or 5xx."""
    time.sleep(RATE_LIMIT_SEC)
    last_exc: Exception = RuntimeError("no attempts made")
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
            print(f"    [retry {attempt}/{retries}] {exc.__class__.__name__} — "
                  f"waiting {wait:.0f}s")
            time.sleep(wait)
    raise last_exc


def extract_fiscal_year(title: str) -> Optional[int]:
    """Infer the fiscal year a filing covers from its title.

    Handles:
      - "Annual Report 2022"        → 2022
      - "Annual Report 2021/22"     → 2022  (slash-short)
      - "Annual Report 2021/2022"   → 2022  (slash-full)
      - "Annual Report 2021-22"     → 2022  (hyphen-short, UK common)
      - "Annual Report 2021-2022"   → 2022  (hyphen-full)
      - "Annual Report"             → None  (no year → caller falls back to release_year)

    Taking the MAX year avoids the ambiguity of "FY starts in YYYY".
    """
    import re
    # YYYY/YY or YYYY-YY  (e.g. 2021/22, 2021-22)
    m = re.search(r'\b(20\d{2})[/-](2\d)\b', title)
    if m:
        return int(m.group(1)) + 1
    # YYYY/YYYY or YYYY-YYYY  (e.g. 2021/2022, 2021-2022)
    m = re.search(r'\b(20\d{2})[/-](20\d{2})\b', title)
    if m:
        return max(int(m.group(1)), int(m.group(2)))
    # Bare 4-digit years in plausible range — take the maximum
    years = [int(y) for y in re.findall(r'\b(20[12]\d)\b', title)
             if 2015 <= int(y) <= 2030]
    return max(years) if years else None


def derive_market_segment(stock_indices: list[dict]) -> str:
    """Classify into FTSE 350, AIM, or Other based on index membership."""
    names = {i.get("name", "") for i in stock_indices}
    if "FTSE 100" in names or "FTSE 250" in names:
        return "FTSE 350"
    if "FTSE AIM All-Share" in names:
        return "AIM"
    return "Other"


def find_in_cache(pk: str) -> Optional[Path]:
    """Return the path to a cached markdown file if any cache dir has it."""
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
    """Return (processing_status, detail) from a non-200 API response."""
    try:
        body = resp.json()
    except ValueError:
        return "unknown", ""
    if not isinstance(body, dict):
        return "unknown", ""
    status = str(body.get("processing_status") or "").strip() or "not_found"
    detail = str(body.get("detail") or "").strip()
    return status, detail


# ── Phase 1: Company list ─────────────────────────────────────────────────────

def phase1_companies(force: bool = False) -> list[dict]:
    out_json = OUT_DIR / "companies.json"
    out_csv  = OUT_DIR / "companies.csv"

    if out_json.exists() and not force:
        print("Phase 1: Loading companies from cache...")
        with open(out_json) as f:
            companies = json.load(f)
        print(f"  {len(companies)} companies loaded.")
        return companies

    print("Phase 1: Fetching all LSE-listed companies from FR API...")
    all_companies: list[dict] = []
    page = 1
    while True:
        r = api_get("/companies/", {
            "listed_stock_exchange": LSE_EXCHANGE_ID,
            "view": "full",
            "page_size": 100,
            "page": page,
        })
        r.raise_for_status()
        d = r.json()
        batch = d.get("results", [])
        all_companies.extend(batch)
        print(f"  page {page}: +{len(batch)} → {len(all_companies)}/{d['count']}", flush=True)
        if not d.get("next"):
            break
        page += 1

    # ── Persist ──────────────────────────────────────────────────────────────
    with open(out_json, "w") as f:
        json.dump(all_companies, f, indent=2)

    csv_fields = [
        "id", "lei", "name", "ticker", "country_code",
        "market_segment", "stock_indices", "listed_exchanges",
        "date_ipo", "is_listed",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for c in all_companies:
            w.writerow({
                "id":               c["id"],
                "lei":              c.get("lei", ""),
                "name":             c["name"],
                "ticker":           c.get("ticker", ""),
                "country_code":     c.get("country_code", ""),
                "market_segment":   derive_market_segment(c.get("stock_index", [])),
                "stock_indices":    "|".join(i["name"] for i in c.get("stock_index", [])),
                "listed_exchanges": "|".join(e["name"] for e in c.get("listed_stock_exchange", [])),
                "date_ipo":         c.get("date_ipo", ""),
                "is_listed":        c.get("is_listed", ""),
            })

    seg_counts = Counter(
        derive_market_segment(c.get("stock_index", [])) for c in all_companies
    )
    print(f"Phase 1 done: {len(all_companies)} companies")
    for seg, cnt in sorted(seg_counts.items(), key=lambda x: -x[1]):
        print(f"  {seg}: {cnt}")
    return all_companies


# ── Phase 2: Filing manifest ──────────────────────────────────────────────────

def select_winner(candidates: list[dict]) -> dict:
    """From filings for the same company+year, pick one.

    Priority: ESEF > regular; tie-break: highest id (most recent submission).
    """
    esef = [f for f in candidates if f.get("filing_type", {}).get("code") == "10-K-ESEF"]
    pool = esef if esef else candidates
    return max(pool, key=lambda f: f["id"])


def phase2_manifest(companies: list[dict]) -> list[dict]:
    checkpoint_path = OUT_DIR / "phase2_checkpoint.json"
    out_json = OUT_DIR / "manifest.json"
    out_csv  = OUT_DIR / "manifest.csv"

    # Load checkpoint
    checkpoint: dict[str, dict] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

    # Load existing manifest entries
    manifest: list[dict] = []
    if out_json.exists():
        with open(out_json) as f:
            manifest = json.load(f)

    done_leis = set(checkpoint.keys())
    pending   = [c for c in companies if c.get("lei") and c["lei"] not in done_leis]
    print(f"Phase 2: {len(done_leis)} companies done, {len(pending)} remaining")

    def save_checkpoint():
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
        with open(out_json, "w") as f:
            json.dump(manifest, f, indent=2)

    for i, company in enumerate(pending, start=1):
        lei     = company["lei"]
        name    = company["name"]
        segment = derive_market_segment(company.get("stock_index", []))

        try:
            r = api_get("/filings/", {
                "lei":      lei,
                "types":    "10-K,10-K-ESEF",
                "ordering": "-release_datetime",
                "page_size": 100,
            })
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            print(f"  [{i}/{len(pending)}] {name}: network error after retries — skipped ({exc})")
            checkpoint[lei] = {"status": "network_error", "years": 0}
            if i % 50 == 0:
                save_checkpoint()
            continue

        if r.status_code != 200:
            print(f"  [{i}/{len(pending)}] {name}: HTTP {r.status_code} — skipped")
            checkpoint[lei] = {"status": f"http_{r.status_code}", "years": 0}
            if i % 50 == 0:
                save_checkpoint()
            continue

        filings = r.json().get("results", [])

        # Group by FISCAL YEAR (inferred from title), filter to >= MIN_YEAR.
        # Falls back to release_year when no year is found in the title.
        # This prevents late-filed reports from being deduped against the
        # following year's report when both land in the same release_year.
        by_fiscal_year: dict[str, list] = defaultdict(list)
        for f in filings:
            release_yr = (f.get("release_datetime") or "")[:4]
            if release_yr < MIN_YEAR:
                continue
            fy = extract_fiscal_year(f.get("title", ""))
            # Clamp: fall back to release_year only when no year is found in
            # the title OR when the inferred FY is implausibly in the future
            # (> 1 year ahead of release).  Do NOT clamp pre-MIN_YEAR fiscal
            # years to release_year — that would force FY2020 reports (filed
            # in 2021) into the FY2021 bucket, wrongly competing with actual
            # FY2021 reports.
            if fy is None or fy > int(release_yr) + 1:
                fiscal_yr = release_yr
            else:
                fiscal_yr = str(fy)
            f["_release_year"] = release_yr   # preserve for the manifest row
            by_fiscal_year[fiscal_yr].append(f)

        # Warn when a bucket holds filings that span multiple release years
        # (confirms a late-filer was correctly separated from its fiscal year).
        for fy_key, candidates in by_fiscal_year.items():
            rel_years = {c["_release_year"] for c in candidates}
            if len(rel_years) > 1:
                print(f"    NOTE: {name} FY{fy_key} has candidates from "
                      f"release_years {sorted(rel_years)} — late filer detected")

        year_entries = []
        for fy_key, candidates in sorted(by_fiscal_year.items()):
            winner      = select_winner(candidates)
            ft          = winner.get("filing_type") or {}
            release_yr  = winner.get("_release_year",
                                     (winner.get("release_datetime") or "")[:4])
            year_entries.append({
                "pk":                str(winner["id"]),
                "company__lei":      lei,
                "company__name":     name,
                "market_segment":    segment,
                "fiscal_year":       fy_key,
                "release_year":      release_yr,
                "release_datetime":  winner.get("release_datetime", ""),
                "title":             winner.get("title", ""),
                "filing_type__code": ft.get("code", ""),
                "filing_type__name": ft.get("name", ""),
                "is_esef":           ft.get("code") == "10-K-ESEF",
                "candidates_count":  len(candidates),
            })

        manifest.extend(year_entries)
        checkpoint[lei] = {"status": "ok", "years": len(year_entries)}

        suffix = f"({len(year_entries)} years)" if year_entries else "(no filings)"
        print(f"  [{i}/{len(pending)}] {name} {suffix}", flush=True)

        if i % 50 == 0 or i == len(pending):
            save_checkpoint()
            print(f"  Checkpoint saved ({len(manifest)} manifest entries total)")

    # Final save + CSV
    save_checkpoint()
    csv_fields = [
        "pk", "company__lei", "company__name", "market_segment",
        "fiscal_year", "release_year", "release_datetime", "title",
        "filing_type__code", "filing_type__name", "is_esef", "candidates_count",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for row in sorted(manifest, key=lambda r: (r["company__lei"], r["fiscal_year"])):
            w.writerow(row)

    unique_cos = len({r["company__lei"] for r in manifest})
    print(f"Phase 2 done: {len(manifest)} entries across {unique_cos} companies")
    fy_counts = Counter(r["fiscal_year"] for r in manifest)
    for fy, cnt in sorted(fy_counts.items()):
        print(f"  FY{fy}: {cnt} filings")
    return manifest


# ── Phase 3: Markdown collection ──────────────────────────────────────────────

def phase3_markdown(manifest: list[dict], dry_run: bool = False) -> list[dict]:
    checkpoint_path = OUT_DIR / "phase3_checkpoint.json"
    new_md_dir      = OUT_DIR / "markdown"
    new_md_dir.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, dict] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

    pending = [r for r in manifest if r["pk"] not in checkpoint]
    cached  = sum(1 for r in manifest if r["pk"] in checkpoint)
    print(f"Phase 3: {cached} already done, {len(pending)} remaining (dry_run={dry_run})")

    def save_checkpoint():
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f)

    # ── Check cache dirs first (no API calls) ────────────────────────────────
    still_pending = []
    for row in pending:
        pk     = row["pk"]
        cached_path = find_in_cache(pk)
        if cached_path:
            checkpoint[pk] = {"status": "cached", "size": cached_path.stat().st_size}
        else:
            still_pending.append(row)

    cache_hits = len(pending) - len(still_pending)
    print(f"  Cache hits: {cache_hits}, API calls needed: {len(still_pending)}")
    save_checkpoint()

    if dry_run:
        print("  Dry-run mode: skipping API calls.")
        still_pending = []

    # ── Fetch remaining from API ──────────────────────────────────────────────
    for i, row in enumerate(still_pending, start=1):
        pk = row["pk"]

        # Also check new_md_dir (previous partial run)
        existing = new_md_dir / f"{pk}.md"
        if existing.exists():
            checkpoint[pk] = {"status": "fetched", "size": existing.stat().st_size}
            continue

        try:
            resp = requests.get(
                f"{API_BASE}/filings/{pk}/markdown/",
                headers=HEADERS,
                timeout=60,
            )
            time.sleep(RATE_LIMIT_SEC)
        except requests.RequestException as exc:
            print(f"  [{i}/{len(still_pending)}] {pk}: request error: {exc}")
            checkpoint[pk] = {"status": "request_error", "detail": str(exc)}
            continue

        if resp.status_code == 200:
            text = extract_markdown_text(resp)
            if text.strip():
                existing.write_text(text, encoding="utf-8")
                checkpoint[pk] = {"status": "fetched", "size": len(text.encode())}
                print(f"  [{i}/{len(still_pending)}] {pk}: fetched ({len(text):,} chars)")
            else:
                checkpoint[pk] = {"status": "empty"}
                print(f"  [{i}/{len(still_pending)}] {pk}: empty response")

        elif resp.status_code == 404:
            proc_status, detail = extract_processing_status(resp)
            checkpoint[pk] = {"status": proc_status, "detail": detail}
            print(f"  [{i}/{len(still_pending)}] {pk}: 404 ({proc_status}){': ' + detail if detail else ''}")

        elif resp.status_code == 403:
            checkpoint[pk] = {"status": "access_denied"}
            print(f"  [{i}/{len(still_pending)}] {pk}: 403 (access denied)")

        else:
            checkpoint[pk] = {"status": f"http_{resp.status_code}"}
            print(f"  [{i}/{len(still_pending)}] {pk}: HTTP {resp.status_code}")

        if i % 50 == 0:
            save_checkpoint()

    save_checkpoint()

    # ── Write processing_status.csv ───────────────────────────────────────────
    status_fields = [
        "pk", "company__lei", "company__name", "market_segment",
        "fiscal_year", "release_year", "filing_type__code", "is_esef",
        "md_status", "md_size", "md_detail",
    ]
    status_rows: list[dict] = []
    for row in manifest:
        pk = row["pk"]
        cp = checkpoint.get(pk, {})
        status_rows.append({
            "pk":                pk,
            "company__lei":      row["company__lei"],
            "company__name":     row["company__name"],
            "market_segment":    row["market_segment"],
            "fiscal_year":       row["fiscal_year"],
            "release_year":      row["release_year"],
            "filing_type__code": row["filing_type__code"],
            "is_esef":           row["is_esef"],
            "md_status":         cp.get("status", "unknown"),
            "md_size":           cp.get("size", ""),
            "md_detail":         cp.get("detail", ""),
        })

    with open(OUT_DIR / "processing_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=status_fields)
        w.writeheader()
        for r in sorted(status_rows, key=lambda x: (x["company__lei"], x["fiscal_year"])):
            w.writerow(r)

    status_counter = Counter(r["md_status"] for r in status_rows)
    print("Phase 3 done. Markdown status breakdown:")
    for status, count in sorted(status_counter.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")

    return status_rows


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(
    companies: list[dict],
    manifest:  list[dict],
    status_rows: list[dict],
) -> None:
    AVAILABLE = {"cached", "fetched"}

    fy_counts  = Counter(r["fiscal_year"]     for r in status_rows if r["md_status"] in AVAILABLE)
    seg_counts = Counter(r["market_segment"]  for r in status_rows if r["md_status"] in AVAILABLE)
    all_status = Counter(r["md_status"]       for r in status_rows)

    summary = {
        "total_companies_in_fr":              len(companies),
        "companies_with_filings_2021_plus":   len({r["company__lei"] for r in manifest}),
        "total_manifest_entries":             len(manifest),
        "markdown_available":                 sum(1 for r in status_rows if r["md_status"] in AVAILABLE),
        "markdown_by_fiscal_year":            dict(sorted(fy_counts.items())),
        "markdown_by_market_segment":         dict(sorted(seg_counts.items())),
        "all_statuses":                       dict(sorted(all_status.items(), key=lambda x: -x[1])),
        "cache_dirs_checked":                 [str(d) for d in CACHE_DIRS],
        "new_markdown_dir":                   str(OUT_DIR / "markdown"),
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=None,
        help="Run only a specific phase (default: all phases)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Phase 3: check cache coverage without making API calls"
    )
    p.add_argument(
        "--refresh-companies", action="store_true",
        help="Phase 1: re-fetch company list even if cached"
    )
    return p.parse_args()


def main() -> None:
    if not API_KEY:
        raise SystemExit("FR_API_KEY not set. Add it to .env.local")

    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_all = args.phase is None

    companies    = None
    manifest     = None
    status_rows  = None

    if run_all or args.phase == 1:
        companies = phase1_companies(force=args.refresh_companies)

    if run_all or args.phase == 2:
        if companies is None:
            companies = phase1_companies()
        manifest = phase2_manifest(companies)

    if run_all or args.phase == 3:
        if manifest is None:
            mf = OUT_DIR / "manifest.json"
            if not mf.exists():
                raise SystemExit("manifest.json not found — run phase 2 first")
            with open(mf) as f:
                manifest = json.load(f)
        status_rows = phase3_markdown(manifest, dry_run=args.dry_run)

    if run_all and companies and manifest and status_rows:
        write_summary(companies, manifest, status_rows)


if __name__ == "__main__":
    main()
