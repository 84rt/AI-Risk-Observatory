#!/usr/bin/env python3
"""################################################################################
Step-by-step pipeline testing with heavy logging for the golden dataset.

Uses filings.xbrl.org API (with LEI codes) as primary source for iXBRL reports,
falls back to Companies House PDF if not available.
################################################################################"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Add pipeline src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.companies_house import CompaniesHouseClient
from src.xbrl_filings_client import XBRLFilingsClient
from src.config import get_settings
from src.database import Database
from src.pdf_extractor import PDFExtractor
from src.ixbrl_extractor import iXBRLExtractor
from src.chunker import TextChunker
from src.llm_classifier import LLMClassifier
from src.aggregator import aggregate_firm

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Load GOLDEN DATASET with LEI codes from companies_with_lei.json
def load_golden_dataset():
    """Load golden dataset with LEI codes."""
    lei_file = Path("data/companies_with_lei.json")
    if lei_file.exists():
        with open(lei_file, 'r') as f:
            return json.load(f)
    else:
        # Fallback to basic dataset without LEI
        logger.warning("LEI file not found. Run lookup_lei_codes.py first for best results.")
        return [
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

GOLDEN_DATASET = load_golden_dataset()


def test_step_1_filing_history():
    """Test Step 1: Fetch filing history and check for available formats."""
    logger.info("=" * 80)
    logger.info("STEP 1: FETCHING FILING HISTORY & CHECKING FORMATS")
    logger.info("=" * 80)
    
    client = CompaniesHouseClient()
    results = {}
    
    for company in tqdm(GOLDEN_DATASET, desc="Fetching filing history"):
        company_number = company["number"]
        company_name = company["name"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {company_name} ({company_number})")
        logger.info(f"{'='*60}")
        
        try:
            # Get filing history
            filings = client.get_filing_history(company_number, category="accounts")
            logger.info(f"✅ Found {len(filings)} total account filings")
            
            # Filter for Annual Accounts (AA type)
            annual_accounts = [
                f for f in filings
                if f.get("type") == "AA" or "annual" in f.get("description", "").lower()
            ]
            
            if not annual_accounts:
                logger.warning(f"⚠️  No annual accounts found for {company_name}")
                results[company_number] = {
                    "status": "no_annual_accounts",
                    "filings_count": len(filings),
                    "annual_accounts_count": 0
                }
                continue
            
            logger.info(f"✅ Found {len(annual_accounts)} annual account filings")
            
            # Get the latest one
            latest = annual_accounts[0]
            logger.info(f"📄 Latest filing:")
            logger.info(f"   Type: {latest.get('type')}")
            logger.info(f"   Description: {latest.get('description')}")
            logger.info(f"   Date: {latest.get('date')}")
            
            # Check available formats
            links = latest.get("links", {})
            available_formats = []
            
            if links.get("document_metadata"):
                available_formats.append("PDF")
            if links.get("xbrl"):
                available_formats.append("XBRL")
            if links.get("ixbrl"):
                available_formats.append("iXBRL")
            
            logger.info(f"📦 Available formats: {', '.join(available_formats) if available_formats else 'None'}")
            logger.info(f"   Links: {list(links.keys())}")
            
            results[company_number] = {
                "status": "success",
                "filings_count": len(filings),
                "annual_accounts_count": len(annual_accounts),
                "latest_filing": {
                    "type": latest.get("type"),
                    "description": latest.get("description"),
                    "date": latest.get("date"),
                    "available_formats": available_formats,
                    "links": list(links.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing {company_name}: {e}", exc_info=True)
            results[company_number] = {
                "status": "error",
                "error": str(e)
            }
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1 SUMMARY")
    logger.info("=" * 80)
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    logger.info(f"✅ Successfully processed: {success_count}/{len(GOLDEN_DATASET)}")
    logger.info(f"⚠️  No annual accounts: {sum(1 for r in results.values() if r.get('status') == 'no_annual_accounts')}")
    logger.info(f"❌ Errors: {sum(1 for r in results.values() if r.get('status') == 'error')}")
    
    return results


def test_step_2_download_documents(year: Optional[int] = None):
    """Test Step 2: Download documents using XBRL API (primary) or Companies House (fallback)."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: DOWNLOADING DOCUMENTS")
    logger.info("Strategy: Try filings.xbrl.org (iXBRL) first, fallback to Companies House (PDF)")
    logger.info("=" * 80)

    xbrl_client = XBRLFilingsClient()
    ch_client = CompaniesHouseClient()
    settings = get_settings()
    output_dir = settings.output_dir / "reports"

    results = {}
    format_counts = {"ixbrl": 0, "pdf": 0}
    source_counts = {"xbrl_api": 0, "companies_house": 0}

    for company in tqdm(GOLDEN_DATASET, desc="Downloading documents"):
        company_number = company["number"]
        company_name = company["name"]
        lei = company.get("lei")

        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading: {company_name} ({company_number})")
        if lei:
            logger.info(f"   LEI: {lei}")
        logger.info(f"{'='*60}")

        result = None

        # Try XBRL API first if we have LEI
        if lei:
            logger.info(f"🔍 Attempting download from filings.xbrl.org...")
            try:
                result = xbrl_client.fetch_annual_report(
                    lei=lei,
                    entity_name=company_name,
                    output_dir=output_dir / "ixbrl",
                    year=year
                )

                if result:
                    logger.info(f"✅ Downloaded from filings.xbrl.org (iXBRL)")
                    source_counts["xbrl_api"] += 1
            except Exception as e:
                logger.warning(f"⚠️  filings.xbrl.org failed: {e}")
                result = None

        # Fallback to Companies House if XBRL failed or no LEI
        if not result:
            if not lei:
                logger.info(f"⚠️  No LEI available, using Companies House...")
            else:
                logger.info(f"📄 Falling back to Companies House...")

            try:
                result = ch_client.fetch_annual_report(
                    company_number=company_number,
                    company_name=company_name,
                    year=year,
                    output_dir=output_dir
                )

                if result:
                    logger.info(f"✅ Downloaded from Companies House")
                    source_counts["companies_house"] += 1
            except Exception as e:
                logger.error(f"❌ Companies House also failed: {e}")
                result = None

        # Process result
        if result and result.get("path") and Path(result["path"]).exists():
            file_path = Path(result["path"])
            file_format = result.get("format", "unknown")
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB

            logger.info(f"   Path: {file_path}")
            logger.info(f"   Size: {file_size:.2f} MB")
            logger.info(f"   Format: {file_format.upper()}")

            format_counts[file_format] = format_counts.get(file_format, 0) + 1

            results[company_number] = {
                "status": "success",
                "path": str(file_path),
                "format": file_format,
                "size_mb": file_size,
                "lei": lei
            }
        else:
            logger.warning(f"⚠️  No document downloaded for {company_name}")
            results[company_number] = {
                "status": "no_document",
                "path": None,
                "lei": lei
            }

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2 SUMMARY")
    logger.info("=" * 80)
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    logger.info(f"✅ Successfully downloaded: {success_count}/{len(GOLDEN_DATASET)}")
    logger.info(f"\nBy format:")
    logger.info(f"   📄 iXBRL/XHTML: {format_counts.get('ixbrl', 0)}")
    logger.info(f"   📄 PDF: {format_counts.get('pdf', 0)}")
    logger.info(f"\nBy source:")
    logger.info(f"   🌐 filings.xbrl.org: {source_counts.get('xbrl_api', 0)}")
    logger.info(f"   🏛️  Companies House: {source_counts.get('companies_house', 0)}")
    logger.info(f"\n⚠️  No documents: {sum(1 for r in results.values() if r.get('status') == 'no_document')}")

    return results


def test_step_3_extract_text():
    """Test Step 3: Extract text from reports (iXBRL/XHTML or PDF)."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: EXTRACTING TEXT FROM REPORTS")
    logger.info("=" * 80)
    
    settings = get_settings()
    reports_dir = settings.output_dir / "reports"
    ixbrl_dir = reports_dir / "ixbrl"
    pdf_dir = reports_dir / "pdfs"
    # Also check legacy pdfs directory
    legacy_pdf_dir = settings.output_dir / "pdfs"
    
    pdf_extractor = PDFExtractor()
    ixbrl_extractor = iXBRLExtractor()
    
    results = {}
    format_counts = {"ixbrl": 0, "pdf": 0}
    
    for company in tqdm(GOLDEN_DATASET, desc="Extracting text"):
        company_number = company["number"]
        company_name = company["name"]
        
        # Find report (prefer iXBRL)
        report_path = None
        report_format = None
        
        if ixbrl_dir.exists():
            matching_ixbrl = list(ixbrl_dir.glob(f"{company_number}_*.xhtml"))
            if matching_ixbrl:
                report_path = matching_ixbrl[0]
                report_format = "ixbrl"
        
        if not report_path:
            matching_pdfs = []
            if pdf_dir.exists():
                matching_pdfs = list(pdf_dir.glob(f"{company_number}_*.pdf"))
            if not matching_pdfs and legacy_pdf_dir.exists():
                matching_pdfs = list(legacy_pdf_dir.glob(f"{company_number}_*.pdf"))
            
            if matching_pdfs:
                report_path = matching_pdfs[0]
                report_format = "pdf"
        
        if not report_path:
            logger.warning(f"⚠️  No report found for {company_name}")
            results[company_number] = {"status": "no_report"}
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting: {company_name}")
        logger.info(f"Format: {report_format.upper()}")
        logger.info(f"File: {report_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            if report_format == "ixbrl":
                extracted = ixbrl_extractor.extract_report(report_path)
                format_counts["ixbrl"] += 1
            else:
                extracted = pdf_extractor.extract_report(report_path)
                format_counts["pdf"] += 1
            
            logger.info(f"✅ Extracted text:")
            if report_format == "pdf":
                logger.info(f"   Pages: {extracted.metadata.get('num_pages', 'N/A')}")
            logger.info(f"   Spans: {extracted.metadata.get('num_spans')}")
            logger.info(f"   Text length: {len(extracted.full_text):,} characters")
            
            # Show sections found
            sections = extracted.sections
            logger.info(f"   Sections found: {len(sections)}")
            for section_name in list(sections.keys())[:5]:  # Show first 5
                logger.info(f"      - {section_name}: {len(sections[section_name])} spans")
            
            results[company_number] = {
                "status": "success",
                "format": report_format,
                "pages": extracted.metadata.get('num_pages'),
                "spans": extracted.metadata.get('num_spans'),
                "text_length": len(extracted.full_text),
                "sections": len(sections)
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting {company_name}: {e}", exc_info=True)
            results[company_number] = {
                "status": "error",
                "error": str(e)
            }
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3 SUMMARY")
    logger.info("=" * 80)
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    logger.info(f"✅ Successfully extracted: {success_count}/{len(GOLDEN_DATASET)}")
    logger.info(f"   📄 iXBRL/XHTML: {format_counts.get('ixbrl', 0)}")
    logger.info(f"   📄 PDF: {format_counts.get('pdf', 0)}")
    logger.info(f"⚠️  No reports: {sum(1 for r in results.values() if r.get('status') == 'no_report')}")
    logger.info(f"❌ Errors: {sum(1 for r in results.values() if r.get('status') == 'error')}")
    
    return results


def test_step_4_chunking():
    """Test Step 4: Chunk extracted text."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: CHUNKING TEXT")
    logger.info("=" * 80)
    
    settings = get_settings()
    reports_dir = settings.output_dir / "reports"
    ixbrl_dir = reports_dir / "ixbrl"
    pdf_dir = reports_dir / "pdfs"
    legacy_pdf_dir = settings.output_dir / "pdfs"
    
    pdf_extractor = PDFExtractor()
    ixbrl_extractor = iXBRLExtractor()
    chunker = TextChunker(chunk_by="paragraph")
    
    results = {}
    total_candidates = 0
    
    for company in tqdm(GOLDEN_DATASET, desc="Chunking text"):
        company_number = company["number"]
        company_name = company["name"]
        
        # Find report (prefer iXBRL)
        report_path = None
        report_format = None
        
        if ixbrl_dir.exists():
            matching_ixbrl = list(ixbrl_dir.glob(f"{company_number}_*.xhtml"))
            if matching_ixbrl:
                report_path = matching_ixbrl[0]
                report_format = "ixbrl"
        
        if not report_path:
            matching_pdfs = []
            if pdf_dir.exists():
                matching_pdfs = list(pdf_dir.glob(f"{company_number}_*.pdf"))
            if not matching_pdfs and legacy_pdf_dir.exists():
                matching_pdfs = list(legacy_pdf_dir.glob(f"{company_number}_*.pdf"))
            
            if matching_pdfs:
                report_path = matching_pdfs[0]
                report_format = "pdf"
        
        if not report_path:
            logger.warning(f"⚠️  No report found for {company_name}")
            results[company_number] = {"status": "no_report"}
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Chunking: {company_name} ({report_format.upper()})")
        logger.info(f"{'='*60}")
        
        try:
            # Extract
            if report_format == "ixbrl":
                extracted = ixbrl_extractor.extract_report(report_path)
            else:
                extracted = pdf_extractor.extract_report(report_path)
            
            # Chunk
            candidates = chunker.chunk_report(
                extracted_report=extracted,
                firm_id=company["ticker"],
                firm_name=company_name,
                sector=company["sector"],
                report_year=2024  # Default year
            )
            
            logger.info(f"✅ Generated {len(candidates)} candidate spans")
            
            if candidates:
                # Show sample
                sample = candidates[0]
                logger.info(f"   Sample span:")
                logger.info(f"      Text length: {len(sample.text)} chars")
                logger.info(f"      Page: {sample.page_number or 'N/A'}")
                logger.info(f"      Section: {sample.section or 'N/A'}")
                logger.info(f"      Preview: {sample.text[:100]}...")
            
            total_candidates += len(candidates)
            results[company_number] = {
                "status": "success",
                "format": report_format,
                "candidates": len(candidates)
            }
            
        except Exception as e:
            logger.error(f"❌ Error chunking {company_name}: {e}", exc_info=True)
            results[company_number] = {
                "status": "error",
                "error": str(e)
            }
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4 SUMMARY")
    logger.info("=" * 80)
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    logger.info(f"✅ Successfully chunked: {success_count}/{len(GOLDEN_DATASET)}")
    logger.info(f"📊 Total candidate spans generated: {total_candidates:,}")
    logger.info(f"⚠️  No reports: {sum(1 for r in results.values() if r.get('status') == 'no_report')}")
    logger.info(f"❌ Errors: {sum(1 for r in results.values() if r.get('status') == 'error')}")
    
    return results


def main():
    """Run step-by-step pipeline tests."""
    logger.info("=" * 80)
    logger.info("AIRO PIPELINE TEST - GOLDEN DATASET (Top 20 UK Companies)")
    logger.info("=" * 80)
    logger.info(f"Testing with {len(GOLDEN_DATASET)} companies")
    logger.info("")
    logger.info("📊 Data Sources:")
    logger.info("   1. filings.xbrl.org - iXBRL/XHTML reports (primary)")
    logger.info("   2. Companies House API - PDF reports (fallback)")
    logger.info("")

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Check if we have LEI codes
    lei_count = sum(1 for c in GOLDEN_DATASET if c.get("lei"))
    logger.info(f"✅ Loaded {lei_count}/{len(GOLDEN_DATASET)} companies with LEI codes")
    if lei_count < len(GOLDEN_DATASET):
        logger.warning(f"⚠️  {len(GOLDEN_DATASET) - lei_count} companies missing LEI - run lookup_lei_codes.py")
    logger.info("")

    # Step 1: Fetch filing history
    step1_results = test_step_1_filing_history()

    # Step 2: Download documents
    step2_results = test_step_2_download_documents(year=2024)

    # Step 3: Extract text
    step3_results = test_step_3_extract_text()

    # Step 4: Chunk text
    step4_results = test_step_4_chunking()
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Step 1 (Filing History): {sum(1 for r in step1_results.values() if r.get('status') == 'success')}/{len(GOLDEN_DATASET)}")
    logger.info(f"Step 2 (Download): {sum(1 for r in step2_results.values() if r.get('status') == 'success')}/{len(GOLDEN_DATASET)}")
    logger.info(f"Step 3 (Extract): {sum(1 for r in step3_results.values() if r.get('status') == 'success')}/{len(GOLDEN_DATASET)}")
    logger.info(f"Step 4 (Chunk): {sum(1 for r in step4_results.values() if r.get('status') == 'success')}/{len(GOLDEN_DATASET)}")
    logger.info("\n✅ Test complete! Check logs/pipeline_test.log for detailed logs.")


if __name__ == "__main__":
    main()

