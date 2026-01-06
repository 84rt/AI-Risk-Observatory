#!/usr/bin/env python3
"""Check if our golden dataset companies are in filings.xbrl.org."""

import requests
import json

# Our golden dataset companies with their LEI codes (need to look these up)
# For now, let's search by company name patterns
GOLDEN_DATASET = [
    ("Shell", "Shell plc"),
    ("AstraZeneca", "AstraZeneca plc"),
    ("HSBC", "HSBC Holdings plc"),
    ("Unilever", "Unilever PLC"),
    ("BP", "BP p.l.c."),
    ("GSK", "GSK plc"),
    ("RELX", "RELX PLC"),
    ("Diageo", "Diageo plc"),
    ("Rio Tinto", "Rio Tinto plc"),
    ("British American Tobacco", "British American Tobacco p.l.c."),
    ("London Stock Exchange", "London Stock Exchange Group plc"),
    ("National Grid", "National Grid plc"),
    ("Compass", "Compass Group PLC"),
    ("Barclays", "Barclays PLC"),
]

base_url = "https://filings.xbrl.org"

print("=" * 80)
print("Checking Golden Dataset Companies in filings.xbrl.org")
print("=" * 80)

found_companies = []

for search_term, full_name in GOLDEN_DATASET:
    print(f"\nSearching for: {full_name}")

    # Get all GB filings and search through them
    # Since the API doesn't support entity name filtering, we need to paginate
    params = {
        "filter[country]": "GB",
        "page[size]": 100
    }

    response = requests.get(f"{base_url}/api/filings", params=params)

    if response.status_code == 200:
        data = response.json()
        filings = data.get("data", [])

        # We need to get entity info to see company names
        # Let's check a few pages
        found = False
        for filing in filings[:100]:  # Check first 100
            entity_link = filing.get("relationships", {}).get("entity", {}).get("links", {}).get("related")

            if entity_link:
                # Get entity details
                entity_response = requests.get(f"{base_url}{entity_link}")
                if entity_response.status_code == 200:
                    entity_data = entity_response.json()
                    entity_name = entity_data.get("data", {}).get("attributes", {}).get("name", "")

                    if search_term.lower() in entity_name.lower():
                        lei = entity_data.get("data", {}).get("id", "")
                        print(f"   ✅ FOUND: {entity_name}")
                        print(f"      LEI: {lei}")
                        print(f"      Latest filing: {filing['attributes']['period_end']}")
                        print(f"      Report URL: {filing['attributes']['report_url']}")

                        found_companies.append({
                            "name": entity_name,
                            "lei": lei,
                            "filing": filing
                        })
                        found = True
                        break

        if not found:
            print(f"   ❌ Not found in first 100 filings")

    else:
        print(f"   Error: {response.status_code}")

print("\n" + "=" * 80)
print(f"Summary: Found {len(found_companies)}/{len(GOLDEN_DATASET)} companies")
print("=" * 80)
