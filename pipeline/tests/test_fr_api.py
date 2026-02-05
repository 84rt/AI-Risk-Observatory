#!/usr/bin/env python3
"""Test FinancialReports.eu API: look up golden set companies and pull filings as markdown.

Usage:
    python pipeline/tests/test_fr_api.py
    python pipeline/tests/test_fr_api.py --company "Shell plc"
    python pipeline/tests/test_fr_api.py --save-markdown
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# ── paths ──────────────────────────────────────────────────────────────────
BASE_REPO = Path(__file__).resolve().parent.parent.parent
PIPELINE_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=BASE_REPO / ".env.local", override=True)

GOLDEN_SET_JSON = BASE_REPO / "data" / "reference" / "golden_set_companies_with_lei.json"
OUTPUT_DIR = BASE_REPO / "data" / "raw" / "fr_markdown"

# ── API config ─────────────────────────────────────────────────────────────
API_BASE = "https://api.financialreports.eu"
API_KEY = os.environ.get("FR_API_KEY")

HEADERS = {
    "x-api-key": API_KEY or "",
    "Accept": "application/json",
}

RATE_LIMIT_PAUSE = 0.5  # seconds between requests


# ── helpers ────────────────────────────────────────────────────────────────
def load_golden_set() -> List[Dict]:
    """Load golden set companies (with LEI codes) from reference JSON."""
    with open(GOLDEN_SET_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def api_get(path: str, params: Optional[Dict] = None) -> requests.Response:
    """Make a GET request to the FR API with rate limiting."""
    url = f"{API_BASE}{path}"
    time.sleep(RATE_LIMIT_PAUSE)
    resp = requests.get(url, headers=HEADERS, params=params or {})
    return resp


# ── core lookups ───────────────────────────────────────────────────────────
def search_company_by_lei(lei: str) -> Optional[Dict]:
    """Search for a company by LEI code. Returns first match or None."""
    resp = api_get("/companies/", params={"lei": lei, "view": "full"})
    if resp.status_code != 200:
        print(f"  /companies/ returned {resp.status_code}: {resp.text[:200]}")
        return None
    data = resp.json()
    results = data.get("results", [])
    return results[0] if results else None


def search_company_by_name(name: str) -> Optional[Dict]:
    """Fallback: search for a company by name."""
    resp = api_get("/companies/", params={"search": name, "view": "full"})
    if resp.status_code != 200:
        print(f"  /companies/ search returned {resp.status_code}: {resp.text[:200]}")
        return None
    data = resp.json()
    results = data.get("results", [])
    return results[0] if results else None


def get_filings_for_company(
    lei: Optional[str] = None,
    company_id: Optional[int] = None,
    types: str = "10-K,10-K-ESEF",
) -> List[Dict]:
    """Get filings for a company, filtered to annual reports by default."""
    params: Dict = {
        "ordering": "-release_datetime",
        "page_size": 10,
        "view": "summary",
    }
    if types:
        params["types"] = types
    if lei:
        params["lei"] = lei
    elif company_id:
        params["company"] = company_id

    resp = api_get("/filings/", params=params)
    if resp.status_code != 200:
        print(f"  /filings/ returned {resp.status_code}: {resp.text[:200]}")
        return []
    data = resp.json()
    return data.get("results", [])


def get_filing_markdown(filing_id: int) -> Optional[str]:
    """Download a filing's markdown content."""
    resp = api_get(f"/filings/{filing_id}/markdown/")
    if resp.status_code == 200:
        # Response may be JSON-wrapped or plain text
        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type:
            body = resp.json()
            # Could be {"markdown": "..."} or just the text
            if isinstance(body, dict):
                return body.get("markdown") or body.get("content") or json.dumps(body, indent=2)
            return str(body)
        return resp.text
    elif resp.status_code == 403:
        print(f"  markdown 403 (Level 2 required): {resp.text[:200]}")
        return None
    elif resp.status_code == 404:
        print(f"  markdown 404 (not processed yet): {resp.text[:200]}")
        return None
    else:
        print(f"  markdown {resp.status_code}: {resp.text[:200]}")
        return None


# ── main test ──────────────────────────────────────────────────────────────
def test_golden_set(
    filter_company: Optional[str] = None,
    save_markdown: bool = False,
) -> None:
    """Look up each golden set company in the FR API and optionally pull markdown."""
    if not API_KEY:
        print("ERROR: FR_API_KEY not set in .env.local")
        sys.exit(1)

    companies = load_golden_set()
    if filter_company:
        companies = [c for c in companies if filter_company.lower() in c["company_name"].lower()]
        if not companies:
            print(f"No golden set company matches '{filter_company}'")
            sys.exit(1)

    if save_markdown:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"FR API key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"Testing {len(companies)} golden set companies against {API_BASE}")
    print("=" * 80)

    results_summary = []

    for company in companies:
        name = company["company_name"]
        lei = company.get("lei")
        sector = company.get("sector", "")
        print(f"\n--- {name} (LEI: {lei}, sector: {sector}) ---")

        # Step 1: Find company
        fr_company = None
        if lei:
            fr_company = search_company_by_lei(lei)
        if not fr_company:
            print(f"  LEI lookup miss, trying name search...")
            fr_company = search_company_by_name(name)

        if fr_company:
            fr_id = fr_company.get("id")
            fr_name = fr_company.get("name", "?")
            fr_country = fr_company.get("country_iso", "?")
            print(f"  FOUND: id={fr_id}, name={fr_name}, country={fr_country}")
        else:
            print(f"  NOT FOUND in FR API")
            results_summary.append({"company": name, "lei": lei, "found": False})
            continue

        # Step 2: Get annual report filings (try LEI first, fall back to company ID)
        filings = get_filings_for_company(lei=lei, company_id=fr_id)
        if not filings and lei:
            print(f"  No filings via LEI, retrying with company id={fr_id}...")
            filings = get_filings_for_company(lei=None, company_id=fr_id)
        print(f"  Annual report filings: {len(filings)}")

        for filing in filings:
            f_id = filing.get("id")
            f_title = filing.get("title", "?")
            f_date = filing.get("release_datetime", "?")
            f_type = filing.get("filing_type", {})
            type_code = f_type.get("code", "?") if isinstance(f_type, dict) else "?"
            print(f"    id={f_id}  type={type_code}  date={f_date[:10] if f_date else '?'}  {f_title[:60]}")

        # Step 3: Try each filing until one has markdown
        md_content = None
        md_filing_id = None
        for filing in filings:
            filing_id = filing["id"]
            md_content = get_filing_markdown(filing_id)
            if md_content:
                md_filing_id = filing_id
                f_title = filing.get("title", "?")
                f_date = filing.get("release_datetime", "?")[:10]
                snippet = md_content[:200].replace("\n", " ")
                print(f"  Markdown OK for filing {filing_id} ({f_date}, {len(md_content)} chars)")
                print(f"    preview: {snippet}...")

                if save_markdown:
                    safe_name = name.replace(" ", "_").replace("/", "_")
                    out_path = OUTPUT_DIR / f"{safe_name}_{filing_id}.md"
                    out_path.write_text(md_content, encoding="utf-8")
                    print(f"    Saved to {out_path}")
                break

        results_summary.append({
            "company": name,
            "lei": lei,
            "found": True,
            "fr_id": fr_company.get("id"),
            "filings_count": len(filings),
            "markdown_available": md_content is not None,
            "markdown_length": len(md_content) if md_content else 0,
            "markdown_filing_id": md_filing_id,
        })

    # ── summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    found = [r for r in results_summary if r.get("found")]
    not_found = [r for r in results_summary if not r.get("found")]
    has_md = [r for r in results_summary if r.get("markdown_available")]

    print(f"Companies found:      {len(found)}/{len(results_summary)}")
    print(f"Companies not found:  {len(not_found)}")
    print(f"Markdown available:   {len(has_md)}/{len(found)}")
    print()

    header = f"{'Company':<35} {'Found':<7} {'FR ID':<8} {'Filings':<8} {'MD':<5} {'MD Size':<10}"
    print(header)
    print("-" * len(header))
    for r in results_summary:
        print(
            f"{r['company']:<35} "
            f"{'yes' if r.get('found') else 'NO':<7} "
            f"{str(r.get('fr_id', '-')):<8} "
            f"{str(r.get('filings_count', '-')):<8} "
            f"{'yes' if r.get('markdown_available') else 'no':<5} "
            f"{r.get('markdown_length', 0):>8}"
        )

    if not_found:
        print("\nMissing companies:")
        for r in not_found:
            print(f"  - {r['company']} (LEI: {r['lei']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test FR API with golden set companies")
    parser.add_argument("--company", type=str, default=None, help="Filter to a single company (substring match)")
    parser.add_argument("--save-markdown", action="store_true", help="Save markdown files to data/raw/fr_markdown/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_golden_set(filter_company=args.company, save_markdown=args.save_markdown)
