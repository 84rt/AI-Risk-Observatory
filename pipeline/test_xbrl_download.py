#!/usr/bin/env python3
"""Test downloading from filings.xbrl.org using LEI codes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.xbrl_filings_client import XBRLFilingsClient

# Test with Shell plc
LEI = "21380068P1DRHMJ8KU70"  # Shell plc
COMPANY_NAME = "Shell plc"

print("=" * 80)
print(f"Testing filings.xbrl.org download for: {COMPANY_NAME}")
print(f"LEI: {LEI}")
print("=" * 80)

client = XBRLFilingsClient()

# Check if entity exists
print("\n1. Checking if entity exists in filings.xbrl.org...")
entity = client.search_entity_by_lei(LEI)

if entity:
    print(f"   ✅ Entity found!")
    attrs = entity.get("attributes", {})
    print(f"   Name: {attrs.get('name')}")
    print(f"   Country: {attrs.get('country')}")
else:
    print(f"   ❌ Entity not found")
    sys.exit(1)

# Get filings
print("\n2. Fetching recent filings...")
filings = client.get_entity_filings(LEI, limit=5)
print(f"   Found {len(filings)} filings")

if filings:
    print("\n   Recent filings:")
    for i, filing in enumerate(filings):
        attrs = filing.get("attributes", {})
        print(f"\n   {i+1}. Period: {attrs.get('period_end')}")
        print(f"      Country: {attrs.get('country')}")
        print(f"      Report URL: {attrs.get('report_url', 'N/A')[:80]}...")
        print(f"      Package URL: {attrs.get('package_url', 'N/A')[:80]}...")

# Download latest
print("\n3. Downloading latest XHTML report...")
output_dir = Path("output/test_xbrl")
result = client.fetch_annual_report(
    lei=LEI,
    entity_name=COMPANY_NAME,
    output_dir=output_dir
)

if result:
    file_path = result["path"]
    file_size = Path(file_path).stat().st_size / 1024  # KB
    print(f"   ✅ Downloaded successfully!")
    print(f"   Path: {file_path}")
    print(f"   Size: {file_size:.1f} KB")
    print(f"   Format: {result['format']}")

    # Show first 500 characters
    print("\n   First 500 characters of XHTML:")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(500)
        print(f"   {content}...")
else:
    print(f"   ❌ Download failed")

print("\n" + "=" * 80)
print("✅ Test complete!")
print("=" * 80)
