#!/usr/bin/env python3
"""
Classify all companies in ch_coverage.csv into:
  Premium      — FTSE 100/250 (definitively Premium Main Market)
  Main Market  — Other Main Market (Premium below FTSE 350, or Standard)
  AIM          — AIM market
  AQSE         — Aquis Stock Exchange
  Other        — Specialist Fund, Sustainable Bond, non-LSE, etc.

Classification logic (in priority order):
  1. FR API stock_indices has FTSE 100 or FTSE 250 → Premium
  2. FR API stock_indices has FTSE AIM All-Share → AIM
  3. listed_exchanges contains "Aquis" → AQSE
  4. LSE instruments API: market = "AIM" → AIM
  5. LSE instruments API: market = "MAINMARKET" → Main Market or Premium (see note)
  6. lse_company_reports_universe.csv market_code:
       ASQ1/ASX1/AMSM → AIM
       SET1/SET2/SET3/STMM/SSMM/SSQ3 → Main Market (Premium or Standard)
       SFM1/SFM2/SSX3/SSX4/SUNM → Other
  7. listed_exchanges is a non-LSE exchange → Other
  8. Unresolved → Other

The Premium/Standard split within Main Market:
  - FTSE 350 (FTSE 100 + FTSE 250) = definitively Premium
  - Main Market with SET1/STMM segment codes (SETS/SETSmm) = likely Premium
  - Main Market with SSMM/SSQ3/SET3 segment codes (SETSqx) = likely Standard
  - But segment codes are trading mechanism proxies, not authoritative listing category
  - Where LSE API confirms segment, we label:
      SET1, STMM → Main Market (Premium likely)
      Others     → Main Market

Usage:
  python scripts/classify_market_segments.py
  python scripts/classify_market_segments.py --no-api      # skip LSE API calls
  python scripts/classify_market_segments.py --dry-run     # print without writing
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from collections import Counter

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

REPO_ROOT = Path(__file__).resolve().parents[1]
CH_COVERAGE   = REPO_ROOT / "data" / "FR_dataset" / "ch_coverage.csv"
COMPANIES_CSV = REPO_ROOT / "data" / "FR_dataset" / "companies.csv"
LSE_UNIVERSE  = REPO_ROOT / "data" / "reference" / "lse_company_reports_universe.csv"
OUTPUT_CSV    = REPO_ROOT / "data" / "reference" / "market_segments.csv"

LSE_API_URL   = "https://api.londonstockexchange.com/api/gw/lse/instruments/alldata/{ticker}"
API_DELAY     = 0.3   # seconds between requests — be polite

# Trading segment codes → AIM vs Main Market
AIM_CODES  = {"ASQ1", "ASX1", "AMSM"}
MAIN_CODES = {"SET1", "SET2", "SET3", "STMM", "SSMM", "SSQ3"}
OTHER_CODES = {"SFM1", "SFM2", "SSX3", "SSX4", "SUNM"}

# Non-LSE exchanges → Other
NON_LSE_EXCHANGES = {
    "Euronext Amsterdam", "Euronext Growth", "Euronext Access",
    "Nasdaq First North Growth Market", "Johannesburg Stock Exchange Limited",
    "Budapest Stock Exchange", "Warsaw Stock Exchange", "CBOE",
    "TSX Venture Exchange",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-api", action="store_true",
                   help="Skip LSE instruments API calls (use cached data only)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print results without writing files")
    p.add_argument("--update-ch-coverage", action="store_true",
                   help="Write market_segment_v2 back into ch_coverage.csv")
    return p.parse_args()


def load_lse_universe() -> dict[str, str]:
    """ticker → market_code"""
    codes: dict[str, str] = {}
    with LSE_UNIVERSE.open() as f:
        for r in csv.DictReader(f):
            codes[r["epic"].strip()] = r["market_code"].strip()
    return codes


def fetch_lse_instrument(ticker: str) -> dict | None:
    """Call LSE API for a single ticker. Returns parsed JSON or None."""
    if not HAS_REQUESTS:
        return None
    url = LSE_API_URL.format(ticker=ticker)
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0 (research)"})
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def classify_from_rules(row: dict, lse_codes: dict[str, str]) -> str:
    """Rule-based classification without API calls."""
    indices  = row.get("stock_indices", "")
    listed   = row.get("listed_exchanges", "")
    ticker   = (row.get("ticker") or "").strip()

    if "FTSE 100" in indices or "FTSE 250" in indices:
        return "Premium"
    if "FTSE AIM All-Share" in indices:
        return "AIM"
    if "Aquis" in listed:
        return "AQSE"

    code = lse_codes.get(ticker, "")
    if code in AIM_CODES:
        return "AIM"
    if code in MAIN_CODES:
        return "Main Market"
    if code in OTHER_CODES:
        return "Other"

    # Check if on a non-LSE exchange
    for ex in NON_LSE_EXCHANGES:
        if ex in listed:
            return "Other"

    return "Unresolved"


def main() -> None:
    args = parse_args()

    lse_codes = load_lse_universe()
    print(f"Loaded LSE universe: {len(lse_codes)} tickers")

    # Load companies.csv for ticker/indices/listed_exchanges
    companies: dict[str, dict] = {}
    with COMPANIES_CSV.open() as f:
        for r in csv.DictReader(f):
            companies[r["lei"]] = r

    # Load ch_coverage.csv
    with CH_COVERAGE.open() as f:
        ch_rows = list(csv.DictReader(f))
    print(f"Loaded ch_coverage: {len(ch_rows)} companies")

    results: list[dict] = []
    api_calls = 0
    api_cache: dict[str, dict] = {}

    unresolved_tickers: list[str] = []

    # Phase 1: rule-based classification
    for ch in ch_rows:
        lei = ch["lei"]
        comp = companies.get(lei, {})

        # Merge useful fields from companies.csv into ch row for classification
        row = {
            "lei": lei,
            "name": ch["name"],
            "stock_indices": comp.get("stock_indices", ""),
            "listed_exchanges": comp.get("listed_exchanges", ch.get("jurisdiction", "")),
            "ticker": comp.get("ticker", ""),
            "market_segment_orig": ch.get("market_segment", ""),
        }

        classification = classify_from_rules(row, lse_codes)
        row["market_segment_v2"] = classification

        if classification == "Unresolved" and row["ticker"] and not args.no_api:
            unresolved_tickers.append(row["ticker"])

        results.append(row)

    phase1_counts = Counter(r["market_segment_v2"] for r in results)
    print("\nPhase 1 (rule-based):")
    for cat in ["Premium", "Main Market", "AIM", "AQSE", "Other", "Unresolved"]:
        print(f"  {cat:<15} {phase1_counts[cat]:>5}")

    # Phase 2: LSE API for unresolved companies
    if unresolved_tickers and not args.no_api:
        print(f"\nPhase 2: calling LSE API for {len(unresolved_tickers)} unresolved companies...")
        for ticker in unresolved_tickers:
            data = fetch_lse_instrument(ticker)
            api_calls += 1
            if data:
                api_cache[ticker] = data
            if api_calls % 25 == 0:
                print(f"  {api_calls}/{len(unresolved_tickers)} done...")
            time.sleep(API_DELAY)

        print(f"  API calls made: {api_calls}, resolved: {len(api_cache)}")

        # Apply API results
        resolved_counts = Counter()
        for row in results:
            if row["market_segment_v2"] != "Unresolved":
                continue
            ticker = row["ticker"]
            data = api_cache.get(ticker)
            if not data:
                row["market_segment_v2"] = "Other"
                resolved_counts["Other (API miss)"] += 1
                continue

            api_market = data.get("market", "")
            api_segment = data.get("segment", "")
            row["api_market"]  = api_market
            row["api_segment"] = api_segment

            if api_market == "AIM":
                row["market_segment_v2"] = "AIM"
                resolved_counts["AIM"] += 1
            elif api_market == "MAINMARKET":
                row["market_segment_v2"] = "Main Market"
                resolved_counts["Main Market"] += 1
            else:
                row["market_segment_v2"] = "Other"
                resolved_counts[f"Other ({api_market or 'unknown'})"] += 1

        print("  Resolved via API:")
        for k, v in resolved_counts.most_common():
            print(f"    {k}: {v}")

        # Remaining unresolved → Other
        for row in results:
            if row["market_segment_v2"] == "Unresolved":
                row["market_segment_v2"] = "Other"

    else:
        # Without API — Unresolved → Other
        for row in results:
            if row["market_segment_v2"] == "Unresolved":
                row["market_segment_v2"] = "Other"

    # Final counts
    final_counts = Counter(r["market_segment_v2"] for r in results)
    total = sum(final_counts.values())
    print(f"\nFinal classification ({total} companies):")
    for cat in ["Premium", "Main Market", "AIM", "AQSE", "Other"]:
        n = final_counts[cat]
        print(f"  {cat:<15} {n:>5}  ({n/total*100:.1f}%)")
    other_cats = [c for c in final_counts if c not in {"Premium","Main Market","AIM","AQSE","Other"}]
    for c in other_cats:
        n = final_counts[c]
        print(f"  {c:<15} {n:>5}  ({n/total*100:.1f}%)")

    if args.dry_run:
        print("\n[dry-run] Not writing files.")
        return

    # Write market_segments.csv
    fields = ["lei", "name", "ticker", "market_segment_orig",
              "market_segment_v2", "stock_indices", "listed_exchanges"]
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote: {OUTPUT_CSV}")

    # Optionally update ch_coverage.csv
    if args.update_ch_coverage:
        seg_map = {r["lei"]: r["market_segment_v2"] for r in results}
        with CH_COVERAGE.open() as f:
            ch_rows2 = list(csv.DictReader(f))
        fieldnames = list(ch_rows2[0].keys())
        if "market_segment" in fieldnames:
            with CH_COVERAGE.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in ch_rows2:
                    row["market_segment"] = seg_map.get(row["lei"], row["market_segment"])
                    writer.writerow(row)
            print(f"Updated: {CH_COVERAGE}")

    # Print segment-code distribution for Main Market (diagnostic)
    main_market_rows = [r for r in results if r.get("market_segment_v2") == "Main Market"]
    if main_market_rows:
        code_dist = Counter(lse_codes.get(r["ticker"], "") for r in main_market_rows)
        print("\nMain Market segment code distribution (LSE trading codes):")
        for code, n in code_dist.most_common():
            print(f"  {code or '(none)':<8} {n}")
        print("  Note: SET1/STMM = likely Premium; SSQ3/SSMM = likely Standard")


if __name__ == "__main__":
    main()
