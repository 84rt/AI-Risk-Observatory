#!/usr/bin/env python3
"""Fetch fiscal year end dates (made_up_date) from Companies House filing history.

Reads ch_coverage.csv (already has company_numbers — no GLEIF needed) and fetches
the `made_up_date` field from each annual accounts filing. This is the definitive
fiscal year end date per filing, zero-cost and direct from the source.

Outputs
-------
  data/FR_dataset/ch_period_of_accounts.csv
    One row per annual filing per company. Columns:
      lei, name, ch_company_number, market_segment,
      submission_date, made_up_date, fiscal_year, filing_type

Usage
-----
  python scripts/fetch_ch_period_of_accounts.py           # full run (resumable)
  python scripts/fetch_ch_period_of_accounts.py --test    # first 20 companies only
  python scripts/fetch_ch_period_of_accounts.py --no-resume  # start fresh

Requires
--------
  CH_API_KEY environment variable
  pip install requests tqdm
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR    = REPO_ROOT / "data" / "FR_dataset"
IN_COVERAGE = DATA_DIR / "ch_coverage.csv"
OUT_POA     = DATA_DIR / "ch_period_of_accounts.csv"
CHECKPOINT  = DATA_DIR / "ch_poa_checkpoint.json"

CH_BASE     = "https://api.company-information.service.gov.uk"
CH_API_KEY  = os.environ.get("CH_API_KEY") or os.environ.get("COMPANIES_HOUSE_API_KEY", "")

ANNUAL_TYPES = {"AA", "AAMD", "AA01", "ACCOUNTS-WITH-ACCOUNTS-EXEMPTION-DATE"}
RATE_CH      = 0.6   # ~1.6 req/s — polite for free tier

# Only keep filings within this fiscal year window
MIN_YEAR = 2019   # a little before 2021 to catch late-filed prior years
MAX_YEAR = 2026


# ── CH API ────────────────────────────────────────────────────────────────────

def ch_filing_history(company_number: str) -> list[dict]:
    """Fetch all accounts filing history pages from Companies House."""
    if not CH_API_KEY:
        raise RuntimeError(
            "CH_API_KEY not set. Get a free key at "
            "https://developer.company-information.service.gov.uk"
        )
    items: list[dict] = []
    start_index = 0
    page_size   = 100
    while True:
        time.sleep(RATE_CH)
        try:
            r = requests.get(
                f"{CH_BASE}/company/{company_number}/filing-history",
                auth=(CH_API_KEY, ""),
                params={
                    "category":       "accounts",
                    "start_index":    start_index,
                    "items_per_page": page_size,
                },
                timeout=20,
            )
        except requests.exceptions.RequestException as e:
            print(f"    Request error for {company_number}: {e}")
            break
        if r.status_code != 200:
            break
        data  = r.json()
        batch = data.get("items", [])
        items.extend(batch)
        total = data.get("total_count", 0)
        start_index += len(batch)
        if start_index >= total or not batch:
            break
    return items


def extract_period_end(item: dict) -> str:
    """Extract made_up_date from a filing history item. Returns '' if absent."""
    dv = item.get("description_values") or {}
    return dv.get("made_up_date", "")


def fiscal_year_from_date(date_str: str) -> int | None:
    """Return the year component of a YYYY-MM-DD date string."""
    if date_str and len(date_str) >= 4:
        try:
            return int(date_str[:4])
        except ValueError:
            pass
    return None


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"done": [], "rows": []}


def save_checkpoint(state: dict) -> None:
    CHECKPOINT.write_text(json.dumps(state, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(resume: bool = True, test_mode: bool = False) -> None:
    if not CH_API_KEY:
        print("ERROR: Set CH_API_KEY environment variable.")
        print("  Get a free key: https://developer.company-information.service.gov.uk")
        sys.exit(1)

    # Load companies — only those we already have a CH number for
    companies = []
    with open(IN_COVERAGE) as f:
        for row in csv.DictReader(f):
            if row["ch_found"] == "True" and row["company_number"]:
                companies.append(row)
    print(f"Loaded {len(companies)} UK-incorporated companies from ch_coverage.csv")

    if test_mode:
        companies = companies[:20]
        print("TEST MODE: processing first 20 companies only")

    state    = load_checkpoint() if resume else {"done": [], "rows": []}
    done_set = set(state["done"])
    rows     = state["rows"]

    todo = [c for c in companies if c["lei"] not in done_set]
    print(f"Remaining: {len(todo)}  (already done: {len(done_set)})")

    for company in tqdm(todo, unit="co"):
        lei    = company["lei"]
        name   = company["name"]
        seg    = company["market_segment"]
        ch_num = company["company_number"]

        items = ch_filing_history(ch_num)

        for item in items:
            if item.get("type") not in ANNUAL_TYPES:
                continue
            made_up_date  = extract_period_end(item)
            submission_dt = item.get("date", "")
            filing_type   = item.get("type", "")

            fy = fiscal_year_from_date(made_up_date)
            if fy is None or not (MIN_YEAR <= fy <= MAX_YEAR):
                continue  # outside our window of interest

            rows.append({
                "lei":              lei,
                "name":             name,
                "ch_company_number": ch_num,
                "market_segment":   seg,
                "submission_date":  submission_dt,
                "made_up_date":     made_up_date,
                "fiscal_year":      fy,
                "filing_type":      filing_type,
            })

        state["done"].append(lei)
        save_checkpoint(state)

    # Write output
    fieldnames = [
        "lei", "name", "ch_company_number", "market_segment",
        "submission_date", "made_up_date", "fiscal_year", "filing_type",
    ]
    with open(OUT_POA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {OUT_POA.relative_to(REPO_ROOT)}")

    # Summary
    from collections import Counter
    year_counts = Counter(r["fiscal_year"] for r in rows)
    lei_counts  = len({r["lei"] for r in rows})
    missing_mud = sum(1 for r in rows if not r["made_up_date"])

    print("\n── Fiscal year breakdown ─────────────────────────")
    for yr in sorted(year_counts):
        print(f"  {yr}:  {year_counts[yr]:4d} filings")
    print(f"\n  Unique companies with filings: {lei_counts}")
    print(f"  Filings with no made_up_date:  {missing_mud}")

    if missing_mud > 0:
        print("\n  WARNING: Some filings have no made_up_date.")
        print("  These rows have fiscal_year inferred from submission_date — treat with caution.")


if __name__ == "__main__":
    resume    = "--no-resume" not in sys.argv
    test_mode = "--test" in sys.argv
    main(resume=resume, test_mode=test_mode)
