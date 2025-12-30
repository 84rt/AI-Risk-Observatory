#!/usr/bin/env python3
"""Test script focused ONLY on downloading documents (iXBRL and PDF)."""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

# Add pipeline src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.companies_house import CompaniesHouseClient
from src.config import get_settings

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/download_test.log')
    ]
)
logger = logging.getLogger(__name__)

# GOLDEN DATASET: Top 20 UK Public Companies
GOLDEN_DATASET = [
    {"rank": 1, "name": "AstraZeneca plc", "number": "02723534", "sector": "Pharmaceuticals", "ticker": "AZN"},
    {"rank": 2, "name": "Shell plc", "number": "04366849", "sector": "Oil & Gas", "ticker": "SHEL"},
    {"rank": 3, "name": "HSBC Holdings plc", "number": "00617987", "sector": "Banking", "ticker": "HSBA"},
    {"rank": 4, "name": "Unilever PLC", "number": "00041424", "sector": "Consumer Goods", "ticker": "ULVR"},
    {"rank": 5, "name": "BP p.l.c.", "number": "00102498", "sector": "Oil & Gas", "ticker": "BP"},
    {"rank": 6, "name": "GSK plc", "number": "03888792", "sector": "Pharmaceuticals", "ticker": "GSK"},
    {"rank": 7, "name": "RELX PLC", "number": "00077536", "sector": "Professional Services", "ticker": "REL"},
    {"rank": 8, "name": "Diageo plc", "number": "00023307", "sector": "Beverages", "ticker": "DGE"},
    {"rank": 9, "name": "Rio Tinto plc", "number": "00719885", "sector": "Mining", "ticker": "RIO"},
    {"rank": 10, "name": "British American Tobacco p.l.c.", "number": "03407696", "sector": "Tobacco", "ticker": "BATS"},
    {"rank": 11, "name": "London Stock Exchange Group plc", "number": "05369106", "sector": "Financial Services", "ticker": "LSEG"},
    {"rank": 12, "name": "National Grid plc", "number": "04031152", "sector": "Utilities", "ticker": "NG"},
    {"rank": 13, "name": "Compass Group PLC", "number": "04083914", "sector": "Food Services", "ticker": "CPG"},
    {"rank": 14, "name": "Barclays PLC", "number": "00048839", "sector": "Banking", "ticker": "BARC"},
    {"rank": 15, "name": "Lloyds Banking Group plc", "number": "SC095000", "sector": "Banking", "ticker": "LLOY"},
    {"rank": 16, "name": "BAE Systems plc", "number": "01470151", "sector": "Aerospace & Defence", "ticker": "BA"},
    {"rank": 17, "name": "Reckitt Benckiser Group plc", "number": "06270876", "sector": "Consumer Goods", "ticker": "RKT"},
    {"rank": 18, "name": "Rolls-Royce Holdings plc", "number": "07524813", "sector": "Aerospace & Defence", "ticker": "RR"},
    {"rank": 19, "name": "Anglo American plc", "number": "03564138", "sector": "Mining", "ticker": "AAL"},
    {"rank": 20, "name": "Tesco PLC", "number": "00445790", "sector": "Retail", "ticker": "TSCO"},
]


def test_downloads_detailed(year: Optional[int] = None):
    """Test downloads with detailed logging of what links are available."""
    logger.info("=" * 80)
    logger.info("DOWNLOAD TEST - Checking Available Formats & Downloading")
    logger.info("=" * 80)
    
    client = CompaniesHouseClient()
    settings = get_settings()
    output_dir = settings.output_dir / "reports"
    
    results = {}
    format_counts = {"ixbrl": 0, "pdf": 0, "failed": 0}
    
    for company in tqdm(GOLDEN_DATASET, desc="Downloading documents"):
        company_number = company["number"]
        company_name = company["name"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {company_name} ({company_number})")
        logger.info(f"{'='*60}")
        
        try:
            # Get filing to inspect available links
            filing = client.get_latest_annual_accounts(company_number, year)
            if not filing:
                logger.warning(f"⚠️  No annual accounts found for {company_name}")
                results[company_number] = {
                    "status": "no_filing",
                    "format": None
                }
                format_counts["failed"] += 1
                continue
            
            links = filing.get("links", {})
            logger.info(f"📋 Available links in filing:")
            for link_name, link_value in links.items():
                logger.info(f"   - {link_name}: {link_value[:80]}..." if len(str(link_value)) > 80 else f"   - {link_name}: {link_value}")
            
            # Check specifically for iXBRL/XBRL
            ixbrl_link = links.get("ixbrl") or links.get("xbrl")
            doc_metadata_link = links.get("document_metadata")
            
            logger.info(f"🔍 Link check:")
            logger.info(f"   - iXBRL/XBRL link: {'✅ Found' if ixbrl_link else '❌ Not found'}")
            logger.info(f"   - PDF metadata link: {'✅ Found' if doc_metadata_link else '❌ Not found'}")
            
            # Try to download
            result = client.fetch_annual_report(
                company_number=company_number,
                company_name=company_name,
                year=year,
                output_dir=output_dir
            )
            
            if result and result.get("path") and Path(result["path"]).exists():
                file_path = Path(result["path"])
                file_format = result.get("format", "unknown")
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                
                logger.info(f"✅ SUCCESS: Downloaded {file_format.upper()}")
                logger.info(f"   Path: {file_path}")
                logger.info(f"   Size: {file_size:.2f} MB")
                
                format_counts[file_format] = format_counts.get(file_format, 0) + 1
                
                results[company_number] = {
                    "status": "success",
                    "format": file_format,
                    "path": str(file_path),
                    "size_mb": file_size,
                    "ixbrl_available": bool(ixbrl_link),
                    "pdf_available": bool(doc_metadata_link)
                }
            else:
                logger.error(f"❌ FAILED: No file downloaded for {company_name}")
                results[company_number] = {
                    "status": "failed",
                    "format": None,
                    "ixbrl_available": bool(ixbrl_link),
                    "pdf_available": bool(doc_metadata_link)
                }
                format_counts["failed"] += 1
                
        except Exception as e:
            logger.error(f"❌ ERROR processing {company_name}: {e}", exc_info=True)
            results[company_number] = {
                "status": "error",
                "error": str(e),
                "format": None
            }
            format_counts["failed"] += 1
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD TEST SUMMARY")
    logger.info("=" * 80)
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    logger.info(f"✅ Successfully downloaded: {success_count}/{len(GOLDEN_DATASET)}")
    logger.info(f"   📄 iXBRL/XHTML: {format_counts.get('ixbrl', 0)}")
    logger.info(f"   📄 PDF (fallback): {format_counts.get('pdf', 0)}")
    logger.info(f"   ❌ Failed: {format_counts.get('failed', 0)}")
    
    # Analyze why iXBRL wasn't downloaded
    ixbrl_available_count = sum(1 for r in results.values() if r.get("ixbrl_available"))
    pdf_available_count = sum(1 for r in results.values() if r.get("pdf_available"))
    
    logger.info(f"\n📊 Format Availability Analysis:")
    logger.info(f"   - Companies with iXBRL link: {ixbrl_available_count}/{len(GOLDEN_DATASET)}")
    logger.info(f"   - Companies with PDF link: {pdf_available_count}/{len(GOLDEN_DATASET)}")
    
    # Show which companies got which format
    logger.info(f"\n📋 Download Results by Company:")
    for company in GOLDEN_DATASET:
        result = results.get(company["number"], {})
        status = result.get("status", "unknown")
        format_type = result.get("format", "N/A")
        ixbrl_avail = result.get("ixbrl_available", False)
        logger.info(f"   {company['name']}: {status} ({format_type}) - iXBRL link: {'✅' if ixbrl_avail else '❌'}")
    
    return results


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("Starting download-only test...")
    logger.info(f"Testing with {len(GOLDEN_DATASET)} companies")
    logger.info("")
    
    results = test_downloads_detailed(year=2024)
    
    logger.info("\n✅ Download test complete!")
    logger.info("Check logs/download_test.log for detailed logs.")










