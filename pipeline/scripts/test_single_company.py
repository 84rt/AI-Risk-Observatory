#!/usr/bin/env python3
"""Test pipeline on a single company for debugging."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import settings
from src.companies_house import CompaniesHouseClient
from src.pdf_extractor import extract_text_from_pdf
from src.chunker import chunk_report
from src.llm_classifier import classify_candidates


def setup_logging(level: str = "INFO"):
    """Set up logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_company(
    company_number: str,
    company_name: str,
    ticker: str,
    sector: str,
    year: int = None,
    max_candidates: int = 10
):
    """Test pipeline on a single company.

    Args:
        company_number: Companies House number
        company_name: Company name
        ticker: Stock ticker
        sector: Sector
        year: Optional year
        max_candidates: Max candidates to classify (for quick testing)
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Testing pipeline on {company_name}")

    # Step 1: Download report
    logger.info("Step 1: Downloading report...")
    client = CompaniesHouseClient()

    output_dir = settings.results_dir / "test_pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = client.fetch_annual_report(
        company_number=company_number,
        company_name=company_name,
        year=year,
        output_dir=output_dir
    )

    if not pdf_path:
        logger.error("Failed to download report")
        return

    logger.info(f"Downloaded to {pdf_path}")

    # Step 2: Extract text
    logger.info("Step 2: Extracting text...")
    extracted = extract_text_from_pdf(pdf_path)
    logger.info(f"Extracted {len(extracted.spans)} spans from {len(extracted.metadata)} pages")

    # Step 3: Chunk
    logger.info("Step 3: Chunking...")
    candidates = chunk_report(
        extracted_report=extracted,
        firm_id=ticker,
        firm_name=company_name,
        sector=sector,
        report_year=year or 2024
    )
    logger.info(f"Generated {len(candidates)} candidates")

    # Limit for testing
    candidates = candidates[:max_candidates]
    logger.info(f"Testing with first {len(candidates)} candidates")

    # Step 4: Classify
    logger.info("Step 4: Classifying with LLM...")
    results = classify_candidates(candidates)

    # Print results
    relevant_count = sum(1 for _, cls in results if cls.is_relevant)
    logger.info(f"\nFound {relevant_count} relevant mentions out of {len(results)}")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for i, (candidate, classification) in enumerate(results, 1):
        if classification.is_relevant:
            print(f"\n[{i}] RELEVANT - {classification.mention_type}")
            print(f"    Risk: {classification.tier_1_category or 'N/A'}")
            print(f"    Specificity: {classification.specificity_level}")
            print(f"    Confidence: {classification.confidence_score:.2f}")
            print(f"    Text: {candidate.text[:200]}...")
            print(f"    Reasoning: {classification.reasoning_summary}")
        else:
            print(f"\n[{i}] Not relevant")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pipeline on single company")

    parser.add_argument("--company-number", required=True, help="Companies House number")
    parser.add_argument("--company-name", required=True, help="Company name")
    parser.add_argument("--ticker", required=True, help="Stock ticker")
    parser.add_argument("--sector", required=True, help="Sector")
    parser.add_argument("--year", type=int, help="Report year")
    parser.add_argument("--max-candidates", type=int, default=10, help="Max candidates to test")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    setup_logging(args.log_level)

    test_company(
        company_number=args.company_number,
        company_name=args.company_name,
        ticker=args.ticker,
        sector=args.sector,
        year=args.year,
        max_candidates=args.max_candidates
    )
