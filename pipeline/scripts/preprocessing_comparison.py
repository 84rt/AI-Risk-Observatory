#!/usr/bin/env python3
"""Test and compare different preprocessing strategies on a single company.

This script helps evaluate which preprocessing approach works best by:
1. Downloading a report
2. Extracting text
3. Applying BOTH preprocessing strategies
4. Saving outputs for comparison
5. Showing statistics
"""

import sys
import json
import logging
from pathlib import Path

# Add pipeline root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.xbrl_filings_client import XBRLFilingsClient
from src.companies_house import CompaniesHouseClient
from src.ixbrl_extractor import iXBRLExtractor
from src.pdf_extractor import PDFExtractor
from src.preprocessor import (
    Preprocessor,
    PreprocessingStrategy,
    preprocess_report
)
from src.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
settings = get_settings()


def load_company_data():
    """Load companies with LEI codes."""
    lei_file = settings.data_dir / "reference" / "companies_with_lei.json"
    if lei_file.exists():
        with open(lei_file, 'r') as f:
            return json.load(f)
    return []


def download_report(company: dict):
    """Download report for a company.

    Returns:
        tuple: (report_path, report_format)
    """
    xbrl_client = XBRLFilingsClient()
    ch_client = CompaniesHouseClient()
    output_dir = settings.raw_dir

    company_name = company["name"]
    company_number = company["number"]
    lei = company.get("lei")

    logger.info(f"\n{'='*70}")
    logger.info(f"Downloading report for: {company_name}")
    logger.info(f"{'='*70}")

    # Try XBRL first
    if lei:
        logger.info("🔍 Trying filings.xbrl.org (iXBRL)...")
        try:
            result = xbrl_client.fetch_annual_report(
                lei=lei,
                entity_name=company_name,
                output_dir=output_dir / "ixbrl"
            )
            if result and result.get("path"):
                logger.info(f"✅ Downloaded iXBRL report")
                return Path(result["path"]), "ixbrl"
        except Exception as e:
            logger.warning(f"⚠️  XBRL download failed: {e}")

    # Fallback to Companies House
    logger.info("📄 Trying Companies House (PDF)...")
    try:
        result = ch_client.fetch_annual_report(
            company_number=company_number,
            company_name=company_name,
            output_dir=output_dir / "pdfs"
        )
        if result and result.get("path"):
            logger.info("✅ Downloaded PDF report")
            return Path(result["path"]), "pdf"
    except Exception as e:
        logger.error(f"❌ PDF download failed: {e}")

    return None, None


def extract_text(report_path: Path, report_format: str):
    """Extract text from report.

    Returns:
        ExtractedReport object
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Extracting text from {report_format.upper()}")
    logger.info(f"{'='*70}")

    if report_format == "ixbrl":
        extractor = iXBRLExtractor()
    else:
        extractor = PDFExtractor()

    extracted = extractor.extract_report(report_path)

    logger.info(f"✅ Extracted:")
    logger.info(f"   Spans: {len(extracted.spans):,}")
    logger.info(f"   Text length: {len(extracted.full_text):,} characters")
    logger.info(f"   Sections: {len(extracted.sections)}")

    return extracted


def compare_strategies(extracted_report, company: dict):
    """Apply both preprocessing strategies and compare results."""
    company_name = company["name"]
    output_dir = settings.processed_dir / "preprocessed"

    logger.info(f"\n{'='*70}")
    logger.info("COMPARING PREPROCESSING STRATEGIES")
    logger.info(f"{'='*70}")

    results = {}

    # Strategy 1: Risk sections only
    logger.info("\n📊 Strategy 1: RISK SECTIONS ONLY")
    logger.info("-" * 70)
    risk_only_preprocessor = Preprocessor(strategy=PreprocessingStrategy.RISK_ONLY)
    risk_only_result = risk_only_preprocessor.process(extracted_report, company_name)

    # Save to file
    output_path_risk = output_dir / "risk_only" / f"{company['number']}_risk_only.md"
    risk_only_preprocessor.save_to_file(risk_only_result, output_path_risk)

    results["risk_only"] = {
        "preprocessed": risk_only_result,
        "output_path": output_path_risk
    }

    # Strategy 2: Keyword-based filtering
    logger.info("\n📊 Strategy 2: KEYWORD-BASED FILTERING")
    logger.info("-" * 70)
    keyword_preprocessor = Preprocessor(
        strategy=PreprocessingStrategy.KEYWORD,
        include_context=True
    )
    keyword_result = keyword_preprocessor.process(extracted_report, company_name)

    # Save to file
    output_path_keyword = output_dir / "keyword" / f"{company['number']}_keyword.md"
    keyword_preprocessor.save_to_file(keyword_result, output_path_keyword)

    results["keyword"] = {
        "preprocessed": keyword_result,
        "output_path": output_path_keyword
    }

    # Print comparison table
    logger.info(f"\n{'='*70}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*70}")

    print("\n┌─────────────────────────┬──────────────┬────────────────┐")
    print("│ Metric                  │ Risk Only    │ Keyword-Based  │")
    print("├─────────────────────────┼──────────────┼────────────────┤")

    def format_cell(value, width=14):
        """Format cell content with proper padding."""
        return f"{str(value):<{width}}"

    # Original spans
    original = len(extracted_report.spans)
    print(f"│ Original spans          │ {format_cell(f'{original:,}')} │ {format_cell(f'{original:,}')} │")

    # Filtered spans
    risk_filtered = risk_only_result.metadata["filtered_spans"]
    keyword_filtered = keyword_result.metadata["filtered_spans"]
    print(f"│ Filtered spans          │ {format_cell(f'{risk_filtered:,}')} │ {format_cell(f'{keyword_filtered:,}')} │")

    # Retention percentage
    risk_pct = risk_only_result.stats["retention_pct"]
    keyword_pct = keyword_result.stats["retention_pct"]
    print(f"│ Retention %             │ {format_cell(f'{risk_pct:.1f}%')} │ {format_cell(f'{keyword_pct:.1f}%')} │")

    # Text length
    risk_len = len(risk_only_result.markdown_content)
    keyword_len = len(keyword_result.markdown_content)
    print(f"│ Markdown length (chars) │ {format_cell(f'{risk_len:,}')} │ {format_cell(f'{keyword_len:,}')} │")

    # Strategy-specific metrics
    if "sections_included" in risk_only_result.stats:
        sections = len(risk_only_result.stats["sections_included"])
        print(f"│ Risk sections found     │ {format_cell(sections)} │ {format_cell('N/A')} │")

    if "ai_keyword_matches" in keyword_result.stats:
        ai_matches = keyword_result.stats["ai_keyword_matches"]
        risk_matches = keyword_result.stats["risk_keyword_matches"]
        print(f"│ AI/ML keyword matches   │ {format_cell('N/A')} │ {format_cell(ai_matches)} │")
        print(f"│ Risk keyword matches    │ {format_cell('N/A')} │ {format_cell(risk_matches)} │")

    print("└─────────────────────────┴──────────────┴────────────────┘")

    # Show sample output
    logger.info("\n📄 Sample outputs (first 500 chars):")
    logger.info("\n--- RISK ONLY ---")
    print(risk_only_result.markdown_content[:500])
    print("...\n")

    logger.info("\n--- KEYWORD-BASED ---")
    print(keyword_result.markdown_content[:500])
    print("...\n")

    logger.info("\n📁 Full outputs saved to:")
    logger.info(f"   Risk only:     {output_path_risk}")
    logger.info(f"   Keyword-based: {output_path_keyword}")

    return results


def main():
    """Main test function."""
    logger.info("=" * 70)
    logger.info("PREPROCESSING STRATEGY COMPARISON TEST")
    logger.info("=" * 70)

    # Load companies
    companies = load_company_data()
    if not companies:
        logger.error("No companies found. Run lookup_lei_codes.py first.")
        return

    # Use Shell plc as test subject (rank 2)
    test_company = next((c for c in companies if c["rank"] == 2), companies[0])

    logger.info(f"\nTest subject: {test_company['name']}")
    logger.info(f"Ticker: {test_company['ticker']}")
    logger.info(f"LEI: {test_company.get('lei', 'N/A')}")

    # Step 1: Download report
    report_path, report_format = download_report(test_company)
    if not report_path:
        logger.error("Failed to download report")
        return

    # Step 2: Extract text
    extracted_report = extract_text(report_path, report_format)

    # Step 3: Compare preprocessing strategies
    results = compare_strategies(extracted_report, test_company)

    logger.info("\n" + "=" * 70)
    logger.info("✅ COMPARISON COMPLETE!")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("1. Review the markdown files in data/processed/preprocessed/")
    logger.info("2. Compare which strategy captures AI risk content better")
    logger.info("3. Choose the strategy that best balances precision and recall")
    logger.info("4. Run full pipeline with chosen strategy")


if __name__ == "__main__":
    main()
