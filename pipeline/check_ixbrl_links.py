#!/usr/bin/env python3
"""Check which companies have direct iXBRL/XBRL links."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.companies_house import CompaniesHouseClient

logging.basicConfig(level=logging.WARNING)

GOLDEN_DATASET = [
    {"rank": 1, "name": "AstraZeneca plc", "number": "02723534"},
    {"rank": 2, "name": "Shell plc", "number": "04366849"},
    {"rank": 3, "name": "HSBC Holdings plc", "number": "00617987"},
    {"rank": 4, "name": "Unilever PLC", "number": "00041424"},
    {"rank": 5, "name": "BP p.l.c.", "number": "00102498"},
    {"rank": 6, "name": "GSK plc", "number": "03888792"},
    {"rank": 7, "name": "RELX PLC", "number": "00077536"},
    {"rank": 8, "name": "Diageo plc", "number": "00023307"},
    {"rank": 9, "name": "Rio Tinto plc", "number": "00719885"},
    {"rank": 10, "name": "British American Tobacco p.l.c.", "number": "03407696"},
    {"rank": 11, "name": "London Stock Exchange Group plc", "number": "05369106"},
    {"rank": 12, "name": "National Grid plc", "number": "04031152"},
    {"rank": 13, "name": "Compass Group PLC", "number": "04083914"},
    {"rank": 14, "name": "Barclays PLC", "number": "00048839"},
    {"rank": 15, "name": "Lloyds Banking Group plc", "number": "SC095000"},
]

def check_ixbrl_links():
    """Check which companies have direct iXBRL/XBRL links."""
    client = CompaniesHouseClient()

    print("=" * 80)
    print(f"{'Company':<45} {'Has iXBRL Link':<20}")
    print("=" * 80)

    ixbrl_count = 0

    for company in GOLDEN_DATASET:
        name = company["name"]
        number = company["number"]

        try:
            # Get filing
            filing = client.get_latest_annual_accounts(number, year=2024)
            if not filing:
                print(f"{name:<45} {'N/A':<20}")
                continue

            links = filing.get("links", {})
            ixbrl_link = links.get("ixbrl") or links.get("xbrl")

            if ixbrl_link:
                ixbrl_count += 1
                print(f"{name:<45} {'✅ YES':<20}")
                print(f"  → {ixbrl_link[:70]}...")
            else:
                print(f"{name:<45} {'❌ NO':<20}")

        except Exception as e:
            print(f"{name:<45} Error: {str(e)[:30]}")

    print("=" * 80)
    print(f"\nSummary: {ixbrl_count}/{len(GOLDEN_DATASET)} companies have iXBRL direct links")

if __name__ == "__main__":
    check_ixbrl_links()
