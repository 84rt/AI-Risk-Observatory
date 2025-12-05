#!/usr/bin/env python3
"""Scan which companies have XHTML/iXBRL format available."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.companies_house import CompaniesHouseClient

# Suppress debug logs
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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
]

def check_available_formats():
    """Check what formats are available for each company."""
    client = CompaniesHouseClient()

    print("=" * 100)
    print(f"{'Company':<40} {'PDF':<8} {'XHTML':<8} {'ZIP':<8}")
    print("=" * 100)

    xhtml_count = 0
    pdf_count = 0
    zip_count = 0

    for company in GOLDEN_DATASET:
        name = company["name"]
        number = company["number"]

        try:
            # Get filing
            filing = client.get_latest_annual_accounts(number, year=2024)
            if not filing:
                print(f"{name:<40} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
                continue

            # Check for direct iXBRL link
            links = filing.get("links", {})
            has_ixbrl_link = bool(links.get("ixbrl") or links.get("xbrl"))

            # Get metadata
            doc_metadata_link = links.get("document_metadata")
            if not doc_metadata_link:
                print(f"{name:<40} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
                continue

            metadata = client._get_document_metadata(doc_metadata_link)
            resources = metadata.get("resources", {})

            has_pdf = "application/pdf" in resources
            has_xhtml = "application/xhtml+xml" in resources
            has_zip = "application/zip" in resources

            # Count
            if has_xhtml:
                xhtml_count += 1
            if has_pdf:
                pdf_count += 1
            if has_zip:
                zip_count += 1

            pdf_mark = "✅" if has_pdf else "❌"
            xhtml_mark = "✅" if has_xhtml else "❌"
            zip_mark = "✅" if has_zip else "❌"
            ixbrl_link_mark = " (direct link)" if has_ixbrl_link else ""

            print(f"{name:<40} {pdf_mark:<8} {xhtml_mark + ixbrl_link_mark:<8} {zip_mark:<8}")

        except Exception as e:
            print(f"{name:<40} Error: {str(e)[:40]}")

    print("=" * 100)
    print(f"\nSummary:")
    print(f"  Companies with XHTML: {xhtml_count}/{len(GOLDEN_DATASET)}")
    print(f"  Companies with PDF: {pdf_count}/{len(GOLDEN_DATASET)}")
    print(f"  Companies with ZIP (ESEF): {zip_count}/{len(GOLDEN_DATASET)}")

if __name__ == "__main__":
    check_available_formats()
