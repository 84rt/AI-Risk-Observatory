#!/usr/bin/env python3
"""Cross-reference LSE companies against Companies House to count expected annual filings.

Pipeline
--------
1. Load 1,469 companies from companies_with_lei.csv (LEI is our key).
2. GLEIF API: LEI → Companies House registration number.
3. Companies House API: registration number → filing history (type=accounts).
4. Count filed annual accounts per company per year → expected_filings.csv.

Output
------
  data/FR_dataset/ch_coverage.csv        — one row per company
  data/FR_dataset/ch_coverage_years.csv  — one row per company × filing year

Usage
-----
  python scripts/fetch_companies_house_coverage.py [--resume]

Requires
--------
  CH_API_KEY environment variable (free key from developer.company-information.service.gov.uk)
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

# ── Config ───────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR    = REPO_ROOT / "data" / "FR_dataset"
INPUT_CSV   = DATA_DIR / "companies_with_lei.csv"
OUT_COMPANY = DATA_DIR / "ch_coverage.csv"
OUT_YEARS   = DATA_DIR / "ch_coverage_years.csv"
CHECKPOINT  = DATA_DIR / "ch_coverage_checkpoint.json"

GLEIF_BASE  = "https://api.gleif.org/api/v1"
CH_BASE     = "https://api.company-information.service.gov.uk"

CH_API_KEY  = os.environ.get("CH_API_KEY") or os.environ.get("COMPANIES_HOUSE_API_KEY", "")

# Annual account filing type codes at Companies House
# AA  = Annual Accounts
# AAMD = Amended Annual Accounts
# AA01 = Dormant company accounts
ANNUAL_TYPES = {"AA", "AAMD", "AA01", "ACCOUNTS-WITH-ACCOUNTS-EXEMPTION-DATE"}

RATE_GLEIF = 0.15   # ~6 req/s (generous limit)
RATE_CH    = 0.6    # ~1.6 req/s (CH free tier allows ~600/min but be polite)


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def gleif_get(lei: str) -> dict | None:
    """Fetch GLEIF record for a LEI. Returns None on failure."""
    time.sleep(RATE_GLEIF)
    try:
        r = requests.get(f"{GLEIF_BASE}/lei-records/{lei}", timeout=20)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.exceptions.RequestException:
        return None


def ch_filing_history(company_number: str) -> list[dict]:
    """Fetch all filing history pages from Companies House for a company.
    Returns list of filing items (dicts). Empty list on failure.
    """
    if not CH_API_KEY:
        raise RuntimeError("CH_API_KEY not set. Get a free key at "
                           "https://developer.company-information.service.gov.uk")
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
                    "category": "accounts",
                    "start_index": start_index,
                    "items_per_page": page_size,
                },
                timeout=20,
            )
        except requests.exceptions.RequestException:
            break
        if r.status_code != 200:
            break
        data   = r.json()
        batch  = data.get("items", [])
        items.extend(batch)
        total  = data.get("total_count", 0)
        start_index += len(batch)
        if start_index >= total or not batch:
            break
    return items


# ── Core logic ───────────────────────────────────────────────────────────────

def registration_number_from_gleif(gleif_data: dict) -> str | None:
    """Extract the local business registry number from a GLEIF record."""
    try:
        reg = gleif_data["data"]["attributes"]["entity"]["registeredAs"]
        return reg if reg else None
    except (KeyError, TypeError):
        pass
    # Older GLEIF schema fallback
    try:
        regs = gleif_data["data"]["attributes"]["entity"]["otherAddresses"]
        for r in regs:
            if r.get("lang") == "en":
                return r.get("addressLines", [None])[0]
    except (KeyError, TypeError):
        pass
    return None


def jurisdiction_from_gleif(gleif_data: dict) -> str | None:
    try:
        return gleif_data["data"]["attributes"]["entity"]["jurisdiction"]
    except (KeyError, TypeError):
        return None


def annual_filing_years(items: list[dict]) -> list[int]:
    """Return sorted list of years in which annual accounts were filed at CH."""
    years = set()
    for item in items:
        if item.get("type") in ANNUAL_TYPES:
            date_str = item.get("date", "")
            if date_str and len(date_str) >= 4:
                try:
                    years.add(int(date_str[:4]))
                except ValueError:
                    pass
    return sorted(years)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"done": [], "results": []}


def save_checkpoint(state: dict) -> None:
    CHECKPOINT.write_text(json.dumps(state, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(resume: bool = True) -> None:
    if not CH_API_KEY:
        print("ERROR: Set CH_API_KEY or COMPANIES_HOUSE_API_KEY environment variable before running.")
        print("  Get a free key: https://developer.company-information.service.gov.uk")
        sys.exit(1)

    # Load company list
    with open(INPUT_CSV) as f:
        companies = list(csv.DictReader(f))
    print(f"Loaded {len(companies)} companies from {INPUT_CSV.name}")

    state = load_checkpoint() if resume else {"done": [], "results": []}
    done_set = set(state["done"])
    results  = state["results"]

    todo = [c for c in companies if c["lei"] not in done_set]
    print(f"Remaining: {len(todo)}  (already done: {len(done_set)})")

    for company in tqdm(todo, unit="co"):
        lei  = company["lei"].strip()
        name = company["name"].strip()
        seg  = company.get("market_segment", "").strip()

        rec = {
            "lei":             lei,
            "name":            name,
            "market_segment":  seg,
            "company_number":  "",
            "jurisdiction":    "",
            "ch_found":        False,
            "annual_filings":  0,
            "filing_years":    "",
            "gleif_status":    "",
            "note":            "",
        }

        # 1. GLEIF lookup
        gleif_data = gleif_get(lei)
        if gleif_data is None:
            rec["gleif_status"] = "error"
            rec["note"] = "GLEIF request failed"
            results.append(rec)
            state["done"].append(lei)
            save_checkpoint(state)
            continue

        reg_num      = registration_number_from_gleif(gleif_data)
        jurisdiction = jurisdiction_from_gleif(gleif_data)
        rec["jurisdiction"] = jurisdiction or ""

        if not reg_num:
            rec["gleif_status"] = "no_reg_num"
            rec["note"] = "GLEIF returned no registration number"
            results.append(rec)
            state["done"].append(lei)
            save_checkpoint(state)
            continue

        rec["gleif_status"]  = "ok"
        rec["company_number"] = reg_num

        # Only query Companies House for UK-registered entities
        if jurisdiction and not jurisdiction.startswith("GB"):
            rec["note"] = f"Non-UK jurisdiction: {jurisdiction}"
            results.append(rec)
            state["done"].append(lei)
            save_checkpoint(state)
            continue

        # 2. Companies House filing history
        items = ch_filing_history(reg_num)
        if not items and jurisdiction and jurisdiction.startswith("GB"):
            # Try zero-padding to 8 chars (CH requires this)
            padded = reg_num.zfill(8)
            if padded != reg_num:
                items = ch_filing_history(padded)
                if items:
                    rec["company_number"] = padded

        years = annual_filing_years(items)
        rec["ch_found"]       = True
        rec["annual_filings"] = len(years)
        rec["filing_years"]   = "|".join(str(y) for y in years)

        results.append(rec)
        state["done"].append(lei)
        save_checkpoint(state)

    # Write outputs
    fields = ["lei", "name", "market_segment", "company_number", "jurisdiction",
              "ch_found", "annual_filings", "filing_years", "gleif_status", "note"]
    with open(OUT_COMPANY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {len(results)} rows → {OUT_COMPANY.relative_to(REPO_ROOT)}")

    # Explode filing years into per-year rows
    year_rows = []
    for r in results:
        for yr_str in r["filing_years"].split("|"):
            if yr_str:
                year_rows.append({
                    "lei":            r["lei"],
                    "name":           r["name"],
                    "market_segment": r["market_segment"],
                    "filing_year":    int(yr_str),
                })
    year_fields = ["lei", "name", "market_segment", "filing_year"]
    with open(OUT_YEARS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=year_fields)
        w.writeheader()
        w.writerows(year_rows)
    print(f"Wrote {len(year_rows)} year-rows → {OUT_YEARS.relative_to(REPO_ROOT)}")

    # Quick summary
    found  = [r for r in results if r["ch_found"]]
    no_reg = [r for r in results if r["gleif_status"] == "no_reg_num"]
    non_uk = [r for r in results if r["note"].startswith("Non-UK")]
    total_expected = sum(r["annual_filings"] for r in found)
    print(f"\n=== Summary ===")
    print(f"  Companies queried:          {len(results)}")
    print(f"  Found at Companies House:   {len(found)}")
    print(f"  No CH reg number (GLEIF):   {len(no_reg)}")
    print(f"  Non-UK jurisdiction:        {len(non_uk)}")
    print(f"  Total expected CH filings:  {total_expected}")
    print(f"  (years 2015-2025 only — filter ch_coverage_years.csv by filing_year)")


if __name__ == "__main__":
    resume = "--no-resume" not in sys.argv
    main(resume=resume)
