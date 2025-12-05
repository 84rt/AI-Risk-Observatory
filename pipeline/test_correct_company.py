#!/usr/bin/env python3
"""Test with correct UK company numbers."""

import sys
sys.path.insert(0, 'src')

import requests
from base64 import b64encode
from src.config import get_settings

settings = get_settings()
api_key = settings.companies_house_api_key

credentials = b64encode(f"{api_key}:".encode()).decode()
headers = {
    "Authorization": f"Basic {credentials}",
    "Accept": "application/json"
}

# Try some well-known UK companies
test_companies = [
    ("Barclays Bank PLC", "01026167"),  # Correct Barclays number
    ("HSBC Holdings plc", "14259"),
    ("BP p.l.c.", "00102498"),  # This one might work
    ("Tesco PLC", "00445790"),
]

print("=" * 60)
print("Testing UK Company Numbers")
print("=" * 60)

for name, number in test_companies:
    print(f"\n{name} ({number}):")
    url = f"https://api.company-information.service.gov.uk/company/{number}"

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Found: {data.get('company_name')}")
            print(f"     Status: {data.get('company_status')}")

            # Try to get filing history
            filing_url = f"https://api.company-information.service.gov.uk/company/{number}/filing-history"
            filing_response = requests.get(filing_url, headers=headers, params={"category": "accounts"})
            if filing_response.status_code == 200:
                filing_data = filing_response.json()
                count = len(filing_data.get('items', []))
                total = filing_data.get('total_count', 0)
                print(f"     Accounts filings: {count} (total: {total})")
        else:
            print(f"  ❌ Not found (status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 60)
print("Use one of the working company numbers above for testing!")
print("=" * 60)
