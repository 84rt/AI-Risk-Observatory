#!/usr/bin/env python3
"""Look up LEI codes for golden dataset companies using GLEIF API."""

import requests
import json
import time
from typing import Optional

from src.config import get_settings

# Golden dataset companies
GOLDEN_DATASET = [
    {"rank": 1, "name": "AstraZeneca plc", "number": "02723534", "ticker": "AZN"},
    {"rank": 2, "name": "Shell plc", "number": "04366849", "ticker": "SHEL"},
    {"rank": 3, "name": "HSBC Holdings plc", "number": "00617987", "ticker": "HSBA"},
    {"rank": 4, "name": "Unilever PLC", "number": "00041424", "ticker": "ULVR"},
    {"rank": 5, "name": "BP p.l.c.", "number": "00102498", "ticker": "BP"},
    {"rank": 6, "name": "GSK plc", "number": "03888792", "ticker": "GSK"},
    {"rank": 7, "name": "RELX PLC", "number": "00077536", "ticker": "REL"},
    {"rank": 8, "name": "Diageo plc", "number": "00023307", "ticker": "DGE"},
    {"rank": 9, "name": "Rio Tinto plc", "number": "00719885", "ticker": "RIO"},
    {"rank": 10, "name": "British American Tobacco p.l.c.", "number": "03407696", "ticker": "BATS"},
    {"rank": 11, "name": "London Stock Exchange Group plc", "number": "05369106", "ticker": "LSEG"},
    {"rank": 12, "name": "National Grid plc", "number": "04031152", "ticker": "NG"},
    {"rank": 13, "name": "Compass Group PLC", "number": "04083914", "ticker": "CPG"},
    {"rank": 14, "name": "Barclays PLC", "number": "00048839", "ticker": "BARC"},
    {"rank": 15, "name": "Lloyds Banking Group plc", "number": "SC095000", "ticker": "LLOY"},
    {"rank": 16, "name": "BAE Systems plc", "number": "01470151", "ticker": "BA"},
    {"rank": 17, "name": "Reckitt Benckiser Group plc", "number": "06270876", "ticker": "RKT"},
    {"rank": 18, "name": "Rolls-Royce Holdings plc", "number": "07524813", "ticker": "RR"},
    {"rank": 19, "name": "Anglo American plc", "number": "03564138", "ticker": "AAL"},
    {"rank": 20, "name": "Tesco PLC", "number": "00445790", "ticker": "TSCO"},
]


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


def lookup_all_lei_codes():
    """Look up LEI codes for all companies in golden dataset."""

    print("=" * 100)
    print("Looking up LEI codes for Golden Dataset Companies")
    print("=" * 100)
    print("\nUsing GLEIF API (Global Legal Entity Identifier Foundation)")
    print()

    results = []

    for company in GOLDEN_DATASET:
        name = company["name"]
        ticker = company["ticker"]

        print(f"{company['rank']:2d}. {name:<45} ({ticker})")

        # Try GLEIF first
        lei, legal_name = search_lei_by_name(name)

        if lei:
            print(f"    ✅ Found via GLEIF: {lei}")
            print(f"       Legal Name: {legal_name}")
            results.append({
                **company,
                "lei": lei,
                "lei_legal_name": legal_name,
                "source": "GLEIF"
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
                    "source": "filings.xbrl.org"
                })
            else:
                print(f"    ❌ Not found")
                results.append({
                    **company,
                    "lei": None,
                    "lei_legal_name": None,
                    "source": None
                })

        # Rate limiting - be nice to APIs
        time.sleep(0.5)
        print()

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    found_count = sum(1 for r in results if r["lei"])
    print(f"\nFound LEI codes for {found_count}/{len(GOLDEN_DATASET)} companies")

    settings = get_settings()
    reference_dir = settings.data_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    output_file = reference_dir / "companies_with_lei.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Generate CSV
    csv_file = reference_dir / "companies_with_lei.csv"
    with open(csv_file, "w") as f:
        f.write("rank,ticker,company_name,company_number,lei,lei_legal_name,sector\n")
        for r in results:
            lei = r.get("lei") or ""
            lei_name = (r.get("lei_legal_name") or "").replace(",", ";")
            sector = r.get("sector", "")
            f.write(f"{r['rank']},{r['ticker']},{r['name']},{r['number']},{lei},\"{lei_name}\",{sector}\n")

    print(f"✅ CSV saved to: {csv_file}")

    # Show companies that need manual lookup
    not_found = [r for r in results if not r["lei"]]
    if not_found:
        print(f"\n⚠️  {len(not_found)} companies need manual LEI lookup:")
        for r in not_found:
            print(f"   - {r['name']} ({r['ticker']})")
            print(f"     Manual lookup: https://search.gleif.org/#/search/{r['name'].replace(' ', '%20')}")


if __name__ == "__main__":
    lookup_all_lei_codes()
