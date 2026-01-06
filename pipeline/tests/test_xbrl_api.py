#!/usr/bin/env python3
"""Test the filings.xbrl.org API to see what UK data is available."""

import requests
import json
from datetime import datetime

def test_xbrl_api():
    """Test filings.xbrl.org API for UK companies."""

    base_url = "https://filings.xbrl.org/api"

    print("=" * 80)
    print("Testing filings.xbrl.org API")
    print("=" * 80)

    # Test 1: Get some UK filings
    print("\n1. Fetching UK filings...")
    params = {
        "filter[filing_system]": "UKSEF",  # UK Single Electronic Format
        "page[size]": 10,
        "sort": "-date"  # Most recent first
    }

    response = requests.get(f"{base_url}/filings", params=params)
    print(f"   Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        filings = data.get("data", [])
        print(f"   Found {len(filings)} filings in response")

        if filings:
            print(f"\n   Sample filings:")
            for i, filing in enumerate(filings[:5]):
                attrs = filing.get("attributes", {})
                print(f"\n   {i+1}. {attrs.get('entity_name', 'N/A')}")
                print(f"      LEI: {attrs.get('lei', 'N/A')}")
                print(f"      Date: {attrs.get('date', 'N/A')}")
                print(f"      Filing System: {attrs.get('filing_system', 'N/A')}")
                print(f"      Package URL: {attrs.get('package_url', 'N/A')[:80]}..." if attrs.get('package_url') else "      Package URL: N/A")
        else:
            print("   No filings found")
    else:
        print(f"   Error: {response.text[:200]}")

    # Test 2: Try to find Shell plc
    print("\n\n2. Searching for Shell plc...")
    params = {
        "filter[entity_name]": "Shell",
        "filter[filing_system]": "UKSEF",
        "page[size]": 5
    }

    response = requests.get(f"{base_url}/filings", params=params)
    if response.status_code == 200:
        data = response.json()
        filings = data.get("data", [])
        print(f"   Found {len(filings)} Shell filings")

        for filing in filings:
            attrs = filing.get("attributes", {})
            print(f"\n   - {attrs.get('entity_name')}")
            print(f"     Date: {attrs.get('date')}")
            print(f"     Package: {attrs.get('package_url', 'N/A')[:80]}...")
    else:
        print(f"   Status: {response.status_code}")

    # Test 3: Check what filing systems are available
    print("\n\n3. Checking available filing systems...")
    response = requests.get(f"{base_url}/filings", params={"page[size]": 100})
    if response.status_code == 200:
        data = response.json()
        filings = data.get("data", [])
        systems = set()
        for filing in filings:
            system = filing.get("attributes", {}).get("filing_system")
            if system:
                systems.add(system)
        print(f"   Filing systems found: {', '.join(sorted(systems))}")

    # Test 4: Get total count of UK filings
    print("\n\n4. Getting total count of UK filings...")
    params = {
        "filter[filing_system]": "UKSEF",
        "page[size]": 1
    }
    response = requests.get(f"{base_url}/filings", params=params)
    if response.status_code == 200:
        data = response.json()
        meta = data.get("meta", {})
        total = meta.get("count", "unknown")
        print(f"   Total UKSEF filings: {total}")

if __name__ == "__main__":
    test_xbrl_api()
