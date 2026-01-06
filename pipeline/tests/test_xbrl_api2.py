#!/usr/bin/env python3
"""Test the filings.xbrl.org API with correct parameters."""

import requests
import json

def test_xbrl_api():
    """Test filings.xbrl.org API."""

    base_url = "https://filings.xbrl.org/api"

    print("=" * 80)
    print("Testing filings.xbrl.org API")
    print("=" * 80)

    # Test 1: Get basic filings without filters
    print("\n1. Fetching recent filings (no filters)...")
    params = {
        "page[size]": 20,
        "sort": "-date"
    }

    response = requests.get(f"{base_url}/filings", params=params)
    print(f"   Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        filings = data.get("data", [])
        print(f"   Found {len(filings)} filings in response")

        # Analyze what we got
        countries = {}
        years = {}
        entities = []

        for filing in filings:
            attrs = filing.get("attributes", {})

            # Look for country indicators
            entity_name = attrs.get("entity_name", "")
            date_str = attrs.get("date", "")
            lei = attrs.get("lei", "")

            # Try to determine country from LEI (first 2 chars after first 2)
            # LEI format: 4 chars (prefix) + 2 chars (country) + rest
            if lei and len(lei) >= 6:
                country_code = lei[4:6]
                countries[country_code] = countries.get(country_code, 0) + 1

            # Extract year
            if date_str:
                year = date_str[:4]
                years[year] = years.get(year, 0) + 1

            entities.append({
                "name": entity_name,
                "date": date_str,
                "lei": lei
            })

        print(f"\n   Country distribution (from LEI):")
        for country, count in sorted(countries.items(), key=lambda x: -x[1])[:10]:
            print(f"      {country}: {count}")

        print(f"\n   Year distribution:")
        for year, count in sorted(years.items(), reverse=True)[:5]:
            print(f"      {year}: {count}")

        print(f"\n   Sample entities:")
        for i, entity in enumerate(entities[:10]):
            print(f"      {i+1}. {entity['name'][:50]} ({entity['date']})")
            if entity['lei']:
                print(f"         LEI: {entity['lei']}")

    else:
        print(f"   Error: {response.text[:500]}")

    # Test 2: Check available fields
    print("\n\n2. Analyzing first filing structure...")
    if response.status_code == 200 and filings:
        first_filing = filings[0]
        print(f"   Available attributes:")
        attrs = first_filing.get("attributes", {})
        for key in sorted(attrs.keys()):
            value = attrs[key]
            if isinstance(value, str) and len(value) > 80:
                value = value[:80] + "..."
            print(f"      {key}: {value}")

    # Test 3: Try entity search
    print("\n\n3. Searching for entities with 'GB' in LEI (UK companies)...")
    response = requests.get(f"{base_url}/filings", params={
        "page[size]": 50,
        "sort": "-date"
    })

    if response.status_code == 200:
        data = response.json()
        filings = data.get("data", [])

        uk_filings = []
        for filing in filings:
            attrs = filing.get("attributes", {})
            lei = attrs.get("lei", "")
            # UK LEI codes have 'GB' in positions 4-6
            if lei and len(lei) >= 6 and lei[4:6] == "GB":
                uk_filings.append({
                    "name": attrs.get("entity_name"),
                    "date": attrs.get("date"),
                    "lei": lei,
                    "package_url": attrs.get("package_url")
                })

        print(f"   Found {len(uk_filings)} UK filings (GB LEI code)")
        for i, filing in enumerate(uk_filings[:10]):
            print(f"\n   {i+1}. {filing['name']}")
            print(f"      Date: {filing['date']}")
            print(f"      LEI: {filing['lei']}")
            print(f"      Package: {filing['package_url'][:80]}..." if filing['package_url'] else "      Package: N/A")

if __name__ == "__main__":
    test_xbrl_api()
