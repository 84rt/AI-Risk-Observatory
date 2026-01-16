#!/usr/bin/env python3
"""Look up LEI codes for golden dataset companies using GLEIF API."""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import requests

from src.company_utils import load_companies_csv
from src.config import get_settings


def search_lei_by_name(company_name: str) -> Optional[str]:
    """Search for LEI code using GLEIF API by company name.

    Args:
        company_name: The company name to search for

    Returns:
        LEI code or None if not found
    """
    # GLEIF (Global Legal Entity Identifier Foundation) API
    base_url = "https://api.gleif.org/api/v1/lei-records"

    # Try searching with the full name
    params = {
        "filter[entity.legalName]": company_name,
        "page[size]": 5
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()
        records = data.get("data", [])

        if records:
            # Return the first match
            lei = records[0].get("id")
            legal_name = records[0].get("attributes", {}).get("entity", {}).get("legalName", {}).get("name", "")
            return lei, legal_name

        return None, None

    except Exception as e:
        print(f"      Error searching GLEIF: {e}")
        return None, None


def search_lei_in_xbrl_filings(company_name: str) -> Optional[str]:
    """Search for company in filings.xbrl.org GB entities.

    Args:
        company_name: The company name to search for

    Returns:
        LEI code or None if not found
    """
    # Get a batch of GB filings and check entity names
    base_url = "https://filings.xbrl.org/api/filings"

    params = {
        "filter[country]": "GB",
        "page[size]": 100,
        "page[number]": 1
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()
        filings = data.get("data", [])

        # Extract unique entity LEIs
        checked_leis = set()

        for filing in filings:
            entity_link = filing.get("relationships", {}).get("entity", {}).get("links", {}).get("related")

            if entity_link:
                entity_id = entity_link.split("/")[-1]

                if entity_id not in checked_leis:
                    checked_leis.add(entity_id)

                    # Get entity details
                    entity_url = f"https://filings.xbrl.org{entity_link}"
                    entity_resp = requests.get(entity_url)

                    if entity_resp.status_code == 200:
                        entity_data = entity_resp.json()
                        entity_name = entity_data.get("data", {}).get("attributes", {}).get("name", "")

                        # Check if name matches (case-insensitive, partial match)
                        search_terms = company_name.lower().replace(" plc", "").replace(" p.l.c.", "").split()
                        entity_name_lower = entity_name.lower()

                        if any(term in entity_name_lower for term in search_terms if len(term) > 3):
                            return entity_id, entity_name

        return None, None

    except Exception as e:
        print(f"      Error searching filings.xbrl.org: {e}")
        return None, None


def lookup_all_lei_codes(companies: list[dict]):
    """Look up LEI codes for companies in a CSV."""

    print("=" * 100)
    print("Looking up LEI codes for Golden Dataset Companies")
    print("=" * 100)
    print("\nUsing GLEIF API (Global Legal Entity Identifier Foundation)")
    print()

    results = []

    for idx, company in enumerate(companies, start=1):
        name = company["company_name"]
        ticker = company.get("ticker") or company.get("company_id")

        print(f"{idx:2d}. {name:<45} ({ticker})")

        if company.get("lei"):
            results.append(
                {
                    **company,
                    "lei_legal_name": None,
                    "lei_source": "existing",
                }
            )
            print(f"    ✅ Existing LEI: {company['lei']}")
            print()
            continue

        # Try GLEIF first
        lei, legal_name = search_lei_by_name(name)

        if lei:
            print(f"    ✅ Found via GLEIF: {lei}")
            print(f"       Legal Name: {legal_name}")
            results.append({
                **company,
                "lei": lei,
                "lei_legal_name": legal_name,
                "lei_source": "GLEIF"
            })
        else:
            # Try filings.xbrl.org
            print(f"    ⚠️  Not found in GLEIF, trying filings.xbrl.org...")
            lei, entity_name = search_lei_in_xbrl_filings(name)

            if lei:
                print(f"    ✅ Found in filings.xbrl.org: {lei}")
                print(f"       Entity Name: {entity_name}")
                results.append({
                    **company,
                    "lei": lei,
                    "lei_legal_name": entity_name,
                    "lei_source": "filings.xbrl.org"
                })
            else:
                print(f"    ❌ Not found")
                results.append({
                    **company,
                    "lei": None,
                    "lei_legal_name": None,
                    "lei_source": None
                })

        # Rate limiting - be nice to APIs
        time.sleep(0.5)
        print()

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    found_count = sum(1 for r in results if r.get("lei"))
    print(f"\nFound LEI codes for {found_count}/{len(companies)} companies")

    settings = get_settings()
    reference_dir = settings.data_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    output_file = reference_dir / "golden_set_companies_with_lei.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Generate CSV
    csv_file = reference_dir / "golden_set_companies_with_lei.csv"
    with open(csv_file, "w") as f:
        f.write("company_id,ticker,company_name,company_number,lei,lei_legal_name,lei_source,sector,index,type\n")
        for r in results:
            lei = r.get("lei") or ""
            lei_name = (r.get("lei_legal_name") or "").replace(",", ";")
            sector = r.get("sector", "")
            index_name = r.get("index", "")
            company_type = r.get("type", "")
            f.write(
                f"{r.get('company_id','')},{r.get('ticker','')},{r.get('company_name','')},"
                f"{r.get('company_number','')},{lei},\"{lei_name}\",{r.get('lei_source','')},"
                f"{sector},{index_name},{company_type}\n"
            )

    print(f"✅ CSV saved to: {csv_file}")

    # Show companies that need manual lookup
    not_found = [r for r in results if not r.get("lei")]
    if not_found:
        print(f"\n⚠️  {len(not_found)} companies need manual LEI lookup:")
        for r in not_found:
            print(f"   - {r['company_name']} ({r.get('ticker') or r.get('company_id')})")
            print(f"     Manual lookup: https://search.gleif.org/#/search/{r['company_name'].replace(' ', '%20')}")

def parse_args() -> argparse.Namespace:
    settings = get_settings()
    default_csv = settings.data_dir / "reference" / "golden_set_companies.csv"
    parser = argparse.ArgumentParser(description="Lookup LEI codes for golden set companies")
    parser.add_argument(
        "--companies",
        type=Path,
        default=default_csv,
        help=f"Path to companies CSV (default: {default_csv})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    companies = load_companies_csv(args.companies)
    lookup_all_lei_codes(companies)
