#!/usr/bin/env python3
"""Check ALL recent filings to find iXBRL ones (not just latest)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.companies_house import CompaniesHouseClient

logging.basicConfig(level=logging.WARNING)

def check_all_recent_filings(company_number: str, company_name: str, limit: int = 10):
    """Check recent filings to find which have iXBRL."""
    client = CompaniesHouseClient()

    print(f"\n{'='*80}")
    print(f"Checking recent filings for: {company_name}")
    print(f"Company Number: {company_number}")
    print(f"{'='*80}\n")

    try:
        # Get ALL recent account filings (not filtered by year)
        filings = client.get_filing_history(company_number, category="accounts")

        print(f"Found {len(filings)} total account filings. Checking first {limit}...\n")

        for i, filing in enumerate(filings[:limit]):
            filing_type = filing.get("type", "N/A")
            description = filing.get("description", "N/A")
            date = filing.get("date", "N/A")

            print(f"{i+1}. Filing from {date}")
            print(f"   Type: {filing_type}")
            print(f"   Description: {description}")

            # Get metadata
            doc_metadata_link = filing.get("links", {}).get("document_metadata")
            if doc_metadata_link:
                try:
                    metadata = client._get_document_metadata(doc_metadata_link)
                    resources = metadata.get("resources", {})

                    formats = list(resources.keys())
                    has_xhtml = "application/xhtml+xml" in resources
                    has_pdf = "application/pdf" in resources
                    has_zip = "application/zip" in resources

                    print(f"   Formats: {', '.join(formats) if formats else 'None listed'}")
                    if has_xhtml:
                        print(f"   ✅ HAS XHTML/iXBRL!")
                        # Show size
                        xhtml_info = resources.get("application/xhtml+xml", {})
                        size = xhtml_info.get("content_length", "unknown")
                        print(f"      Size: {size:,} bytes" if isinstance(size, int) else f"      Size: {size}")
                    elif has_zip:
                        print(f"   ✅ HAS ZIP (ESEF package)!")
                    elif has_pdf:
                        print(f"   📄 Only PDF available")
                    else:
                        print(f"   ⚠️ No formats listed")

                except Exception as e:
                    print(f"   ❌ Error getting metadata: {e}")
            else:
                print(f"   ⚠️ No document metadata link")

            print()

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Check a few representative companies
    companies = [
        ("04366849", "Shell plc"),
        ("02723534", "AstraZeneca plc"),
        ("00048839", "Barclays PLC"),
    ]

    for number, name in companies:
        check_all_recent_filings(number, name, limit=5)
