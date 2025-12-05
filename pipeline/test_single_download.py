#!/usr/bin/env python3
"""Test downloading a single company's annual report with detailed logging."""

import logging
import sys
from pathlib import Path

# Add pipeline src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.companies_house import CompaniesHouseClient
from src.config import get_settings

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_single_company(company_number: str, company_name: str):
    """Test downloading a single company's annual report.

    Args:
        company_number: The Companies House number
        company_name: The company name
    """
    logger.info("=" * 80)
    logger.info(f"Testing Download for: {company_name}")
    logger.info(f"Company Number: {company_number}")
    logger.info("=" * 80)

    settings = get_settings()
    output_dir = settings.output_dir / "reports"

    client = CompaniesHouseClient()

    try:
        # Step 1: Get filing history
        logger.info("\nStep 1: Fetching filing history...")
        filing = client.get_latest_annual_accounts(company_number, year=2024)

        if not filing:
            logger.error(f"❌ No annual accounts found for {company_name}")
            return

        logger.info(f"✅ Found filing:")
        logger.info(f"   Type: {filing.get('type')}")
        logger.info(f"   Description: {filing.get('description')}")
        logger.info(f"   Date: {filing.get('date')}")

        # Step 2: Check available links
        links = filing.get("links", {})
        logger.info(f"\nStep 2: Available links in filing:")
        for key, value in links.items():
            logger.info(f"   - {key}: {value[:80]}..." if len(str(value)) > 80 else f"   - {key}: {value}")

        # Step 3: Download
        logger.info("\nStep 3: Attempting download...")
        result = client.fetch_annual_report(
            company_number=company_number,
            company_name=company_name,
            year=2024,
            output_dir=output_dir
        )

        if result and result.get("path"):
            file_path = Path(result["path"])
            file_format = result.get("format")
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB

            logger.info(f"\n{'='*80}")
            logger.info(f"✅ SUCCESS!")
            logger.info(f"{'='*80}")
            logger.info(f"Format: {file_format.upper()}")
            logger.info(f"Path: {file_path}")
            logger.info(f"Size: {file_size:.2f} MB")

            # Show first few lines if it's XHTML
            if file_format == "ixbrl" and file_path.suffix in ['.xhtml', '.html']:
                logger.info(f"\nFirst 500 characters of XHTML:")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(500)
                    logger.info(f"{content}...")
        else:
            logger.error(f"❌ Failed to download document")

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)


if __name__ == "__main__":
    # Test with Shell plc (known to have iXBRL)
    test_single_company(
        company_number="04366849",
        company_name="Shell plc"
    )
