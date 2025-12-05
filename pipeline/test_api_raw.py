#!/usr/bin/env python3
"""Debug Companies House API at HTTP level."""

import sys
sys.path.insert(0, 'src')

import requests
from base64 import b64encode
from src.config import get_settings

settings = get_settings()
api_key = settings.companies_house_api_key

print("=" * 60)
print("Testing Companies House API (Raw HTTP)")
print("=" * 60)
print(f"API Key: {api_key[:20]}...")
print()

# Basic Auth with API key as username, empty password
credentials = b64encode(f"{api_key}:".encode()).decode()

headers = {
    "Authorization": f"Basic {credentials}",
    "Accept": "application/json"
}

# Test company profile first
company_number = "00489800"
url = f"https://api.company-information.service.gov.uk/company/{company_number}"

print(f"Testing URL: {url}")
print()

try:
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    print()

    if response.status_code == 200:
        data = response.json()
        print("✅ Company found!")
        print(f"Company Name: {data.get('company_name')}")
        print(f"Company Status: {data.get('company_status')}")
        print(f"Company Type: {data.get('type')}")
    else:
        print("❌ Failed to get company")
        print(f"Response: {response.text}")

    # Now try filing history
    print("\n" + "=" * 60)
    print("Testing Filing History...")
    print("=" * 60)

    filing_url = f"https://api.company-information.service.gov.uk/company/{company_number}/filing-history"
    response = requests.get(filing_url, headers=headers)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        items = data.get('items', [])
        print(f"✅ Found {len(items)} filings")
        print(f"Total count: {data.get('total_count', 0)}")

        if items:
            print("\nFirst filing:")
            print(items[0])
    else:
        print("❌ Failed to get filings")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()
