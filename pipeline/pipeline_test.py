#!/usr/bin/env python3
"""Step-by-step pipeline testing with heavy logging for the golden dataset."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Add pipeline src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.companies_house import CompaniesHouseClient
from src.config import get_settings
from src.database import Database
from src.pdf_extractor import PDFExtractor
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
    """Test Step 2: Download documents (PDFs for now, iXBRL later)."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: DOWNLOADING DOCUMENTS")
    logger.info("=" * 80)
    
    client = CompaniesHouseClient()
    settings = get_settings()
    output_dir = settings.output_dir / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for company in tqdm(GOLDEN_DATASET, desc="Downloading documents"):
        company_number = company["number"]
        company_name = company["name"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading: {company_name} ({company_number})")
        logger.info(f"{'='*60}")
        
        try:
            pdf_path = client.fetch_annual_report(
                company_number=company_number,
                company_name=company_name,
                year=year,
                output_dir=output_dir
            )
            
            if pdf_path and Path(pdf_path).exists():
                file_size = Path(pdf_path).stat().st_size / (1024 * 1024)  # MB
                logger.info(f"✅ Downloaded: {pdf_path}")
                logger.info(f"   Size: {file_size:.2f} MB")
                results[company_number] = {
                    "status": "success",
                    "path": str(pdf_path),
                    "size_mb": file_size
                }
            else:
                logger.warning(f"⚠️  No document downloaded for {company_name}")
                results[company_number] = {
                    "status": "no_document",
                    "path": None
                }
                
        except Exception as e:
            logger.error(f"❌ Error downloading {company_name}: {e}", exc_info=True)
            results[company_number] = {
                "status": "error",
                "error": str(e)
            }
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2 SUMMARY")
    logger.info("=" * 80)
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    logger.info(f"✅ Successfully downloaded: {success_count}/{len(GOLDEN_DATASET)}")
    logger.info(f"⚠️  No documents: {sum(1 for r in results.values() if r.get('status') == 'no_document')}")
    logger.info(f"❌ Errors: {sum(1 for r in results.values() if r.get('status') == 'error')}")
    
    return results


def test_step_3_extract_text():
    """Test Step 3: Extract text from PDFs."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: EXTRACTING TEXT FROM PDFS")
    logger.info("=" * 80)
    
    settings = get_settings()
    pdf_dir = settings.output_dir / "pdfs"
    extractor = PDFExtractor()
    
    results = {}
    
    for company in tqdm(GOLDEN_DATASET, desc="Extracting text"):
        company_number = company["number"]
        company_name = company["name"]
        
        # Find PDF
        matching_pdfs = list(pdf_dir.glob(f"{company_number}_*.pdf"))
        
        if not matching_pdfs:
            logger.warning(f"⚠️  No PDF found for {company_name}")
            results[company_number] = {"status": "no_pdf"}
            continue
        
        pdf_path = matching_pdfs[0]
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting: {company_name}")
        logger.info(f"PDF: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            extracted = extractor.extract_report(pdf_path)
            
            logger.info(f"✅ Extracted text:")
            logger.info(f"   Pages: {extracted.metadata.get('num_pages')}")
            logger.info(f"   Spans: {extracted.metadata.get('num_spans')}")
            logger.info(f"   Text length: {len(extracted.full_text):,} characters")
            
            # Show sections found
            sections = extracted.sections
            logger.info(f"   Sections found: {len(sections)}")
            for section_name in list(sections.keys())[:5]:  # Show first 5
                logger.info(f"      - {section_name}: {len(sections[section_name])} spans")
            
            results[company_number] = {
                "status": "success",
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
    logger.info(f"⚠️  No PDFs: {sum(1 for r in results.values() if r.get('status') == 'no_pdf')}")
    logger.info(f"❌ Errors: {sum(1 for r in results.values() if r.get('status') == 'error')}")
    
    return results


def test_step_4_chunking():
    """Test Step 4: Chunk extracted text."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: CHUNKING TEXT")
    logger.info("=" * 80)
    
    settings = get_settings()
    pdf_dir = settings.output_dir / "pdfs"
    extractor = PDFExtractor()
    chunker = TextChunker()
    
    results = {}
    total_candidates = 0
    
    for company in tqdm(GOLDEN_DATASET, desc="Chunking text"):
        company_number = company["number"]
        company_name = company["name"]
        
        # Find PDF
        matching_pdfs = list(pdf_dir.glob(f"{company_number}_*.pdf"))
        
        if not matching_pdfs:
            logger.warning(f"⚠️  No PDF found for {company_name}")
            results[company_number] = {"status": "no_pdf"}
            continue
        
        pdf_path = matching_pdfs[0]
        logger.info(f"\n{'='*60}")
        logger.info(f"Chunking: {company_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Extract
            extracted = extractor.extract_report(pdf_path)
            
            # Chunk
            candidates = chunker.chunk_report(
                extracted_report=extracted,
                firm_id=company["ticker"],
                firm_name=company_name,
                sector=company["sector"],
                report_year=2024,  # Default year
                chunk_by="paragraph"
            )
            
            logger.info(f"✅ Generated {len(candidates)} candidate spans")
            
            if candidates:
                # Show sample
                sample = candidates[0]
                logger.info(f"   Sample span:")
                logger.info(f"      Text length: {len(sample.text)} chars")
                logger.info(f"      Page: {sample.page_number}")
                logger.info(f"      Section: {sample.section or 'N/A'}")
                logger.info(f"      Preview: {sample.text[:100]}...")
            
            total_candidates += len(candidates)
            results[company_number] = {
                "status": "success",
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
    logger.info(f"⚠️  No PDFs: {sum(1 for r in results.values() if r.get('status') == 'no_pdf')}")
    logger.info(f"❌ Errors: {sum(1 for r in results.values() if r.get('status') == 'error')}")
    
    return results


def main():
    """Run step-by-step pipeline tests."""
    logger.info("=" * 80)
    logger.info("AIRO PIPELINE TEST - GOLDEN DATASET (Top 20 UK Companies)")
    logger.info("=" * 80)
    logger.info(f"Testing with {len(GOLDEN_DATASET)} companies")
    logger.info("")
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Step 1: Fetch filing history
    step1_results = test_step_1_filing_history()
    
    # Step 2: Download documents
    step2_results = test_step_2_download_documents(year=2024)
    
    # Step 3: Extract text (only for successfully downloaded PDFs)
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

