#!/usr/bin/env python3
"""Quick test of Companies House API."""

import sys
sys.path.insert(0, 'src')

from src.companies_house import CompaniesHouseClient
from src.config import get_settings

# Test config
print("=" * 60)
print("Testing Configuration...")
print("=" * 60)
settings = get_settings()
print(f"✅ Gemini Model: {settings.gemini_model}")
print(f"✅ Gemini Key: {settings.gemini_api_key[:20]}...")
print(f"✅ CH Key: {settings.companies_house_api_key[:20]}...")

# Test Companies House API
print("\n" + "=" * 60)
print("Testing Companies House API...")
print("=" * 60)

client = CompaniesHouseClient()

# Try Barclays
print("\nFetching filings for Barclays (00489800)...")
try:
    filings = client.get_filing_history("00489800")
    print(f"✅ Found {len(filings)} total filings")

    if filings:
        print("\nFirst 5 filings:")
        for i, filing in enumerate(filings[:5], 1):
            print(f"\n{i}. Type: {filing.get('type')} | Category: {filing.get('category')}")
            print(f"   Description: {filing.get('description')[:80]}...")
            print(f"   Date: {filing.get('date')}")

    # Filter for accounts
    accounts_filings = client.get_filing_history("00489800", category="accounts")
    print(f"\n✅ Found {len(accounts_filings)} accounts filings")

    if accounts_filings:
        print("\nFirst 3 accounts filings:")
        for i, filing in enumerate(accounts_filings[:3], 1):
            print(f"\n{i}. Type: {filing.get('type')}")
            print(f"   Description: {filing.get('description')}")
            print(f"   Date: {filing.get('date')}")
            has_doc = filing.get('links', {}).get('document_metadata')
            print(f"   Has document: {'✅' if has_doc else '❌'}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
