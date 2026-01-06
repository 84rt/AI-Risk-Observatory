#!/usr/bin/env python3
"""
Comprehensive Classifier Test Suite for AIRO Pipeline.

This is the main test orchestrator that:
1. Creates a run_id and logs configuration
2. Loads preprocessed files for each company/year
3. Runs each classifier sequentially
4. Stores all results to database
5. Exports annotation files
6. Generates summary report

Usage:
    python tests/run_comprehensive_tests.py [--years 2023 2024] [--classifiers harms adoption]
"""

import argparse
import csv
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add pipeline root to path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifiers import (
    HarmsClassifier,
    AdoptionTypeClassifier,
    SubstantivenessClassifier,
    VendorClassifier,
    RiskClassifier,
)
from src.database import Database, get_database
from src.utils.logging_config import setup_logging, log_run_summary
from src.utils.data_export import DataExporter, export_run_metadata
from src.config import get_settings

# Paths
settings = get_settings()
DATA_DIR = settings.data_dir
PREPROCESSED_DIR = settings.processed_dir / "preprocessed" / "keyword"
COMPANIES_CSV = settings.data_dir / "reference" / "companies_with_lei.csv"

# Available classifiers
AVAILABLE_CLASSIFIERS = {
    "harms": HarmsClassifier,
    "adoption": AdoptionTypeClassifier,
    "substantiveness": SubstantivenessClassifier,
    "vendor": VendorClassifier,
    "risk": RiskClassifier,
}


def load_companies(csv_path: Path) -> List[Dict[str, str]]:
    """Load company list from CSV.

    Args:
        csv_path: Path to companies CSV file

    Returns:
        List of company dicts with ticker, company_number, company_name, etc.
    """
    companies = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            companies.append({
                "ticker": row.get("ticker", ""),
                "company_number": row.get("company_number", ""),
                "company_name": row.get("company_name", ""),
                "lei": row.get("lei", ""),
                "sector": row.get("sector", "Unknown"),
            })
    return companies


def find_preprocessed_files(
    company_number: str,
    ticker: str,
    years: List[int]
) -> Dict[int, Path]:
    """Find preprocessed files for a company and years.

    Args:
        company_number: Companies House number
        ticker: Company ticker symbol
        years: List of years to find

    Returns:
        Dict mapping year to file path
    """
    found_files = {}

    for year in years:
        # Try different filename patterns
        patterns = [
            f"{company_number}_{ticker}_{year}.md",
            f"{company_number}_{ticker.upper()}_{year}.md",
        ]

        for pattern in patterns:
            file_path = PREPROCESSED_DIR / pattern
            if file_path.exists():
                found_files[year] = file_path
                break

    return found_files


def generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"


def run_classifier(
    classifier_class,
    run_id: str,
    company: Dict[str, str],
    file_path: Path,
    year: int,
) -> Dict[str, Any]:
    """Run a single classifier on a single file.

    Args:
        classifier_class: The classifier class to instantiate
        run_id: Run ID
        company: Company dict
        file_path: Path to preprocessed file
        year: Report year

    Returns:
        Classification result as dict
    """
    classifier = classifier_class(run_id=run_id)

    result = classifier.classify_report(
        report_path=file_path,
        firm_id=company["ticker"],
        firm_name=company["company_name"],
        report_year=year,
        sector=company["sector"],
    )

    return result.to_dict()


def run_full_test_suite(
    companies_csv: Path = COMPANIES_CSV,
    years: List[int] = [2023, 2024],
    classifiers: List[str] = ["harms", "adoption", "substantiveness"],
    model: str = "gemini-2.0-flash",
    rate_limit_delay: float = 1.0,
    save_to_db: bool = True,
    export_results: bool = True,
) -> str:
    """
    Run the full classifier test suite.

    Args:
        companies_csv: Path to companies CSV file
        years: List of years to process
        classifiers: List of classifier names to run
        model: Model to use
        rate_limit_delay: Delay between API calls
        save_to_db: Whether to save results to database
        export_results: Whether to export annotation files

    Returns:
        run_id
    """
    # Generate run ID
    run_id = generate_run_id()

    # Set up logging
    logger = setup_logging(log_level="INFO", run_id=run_id)
    logger.info(f"Starting comprehensive test suite: {run_id}")

    # Configuration
    config = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "years": years,
        "classifiers": classifiers,
        "model": model,
        "rate_limit_delay": rate_limit_delay,
    }

    # Load companies
    companies = load_companies(companies_csv)
    config["companies"] = len(companies)
    logger.info(f"Loaded {len(companies)} companies from {companies_csv}")

    # Initialize database
    db = get_database() if save_to_db else None

    # Create run record
    if db:
        db.create_run(run_id, config, model)

    # Track results
    all_results = []
    start_time = time.time()
    success_count = 0
    error_count = 0
    confidences = []

    # Process each company
    for company_idx, company in enumerate(companies, 1):
        company_name = company["company_name"]
        company_number = company["company_number"]
        ticker = company["ticker"]

        logger.info(f"\n[{company_idx}/{len(companies)}] Processing: {company_name}")

        # Find preprocessed files
        files = find_preprocessed_files(company_number, ticker, years)

        if not files:
            logger.warning(f"  No preprocessed files found for {company_name}")
            continue

        # Process each year
        for year, file_path in sorted(files.items()):
            logger.info(f"  Year {year}: {file_path.name}")

            # Run each classifier
            for clf_name in classifiers:
                if clf_name not in AVAILABLE_CLASSIFIERS:
                    logger.warning(f"  Unknown classifier: {clf_name}")
                    continue

                classifier_class = AVAILABLE_CLASSIFIERS[clf_name]
                logger.info(f"    Running {clf_name} classifier...")

                try:
                    result = run_classifier(
                        classifier_class=classifier_class,
                        run_id=run_id,
                        company=company,
                        file_path=file_path,
                        year=year,
                    )

                    if result["success"]:
                        success_count += 1
                        confidences.append(result["confidence_score"])
                        logger.info(
                            f"      Result: {result['primary_label']} "
                            f"(confidence={result['confidence_score']:.2f})"
                        )
                    else:
                        error_count += 1
                        logger.error(f"      Error: {result.get('error_message')}")

                    all_results.append(result)

                    # Save to database
                    if db:
                        db.save_classification_result(result)

                        # Save evidence snippets
                        evidence = result.get("evidence", [])
                        for i, excerpt in enumerate(evidence):
                            snippet_data = {
                                "snippet_id": f"{result['result_id']}_snippet_{i}",
                                "result_id": result["result_id"],
                                "text_excerpt": excerpt[:1000] if isinstance(excerpt, str) else str(excerpt)[:1000],
                                "source_file": str(file_path),
                                "category": clf_name,
                                "needs_review": result["confidence_score"] < 0.7,
                                "review_priority": 1 if result["confidence_score"] < 0.7 else 3,
                            }
                            db.save_evidence_snippet(snippet_data)

                except Exception as e:
                    logger.error(f"      Exception: {e}")
                    error_count += 1
                    all_results.append({
                        "result_id": f"{run_id}_{clf_name}_{ticker}_{year}_error",
                        "run_id": run_id,
                        "firm_id": ticker,
                        "firm_name": company_name,
                        "report_year": year,
                        "classifier_type": clf_name,
                        "success": False,
                        "error_message": str(e),
                    })

                # Rate limiting
                time.sleep(rate_limit_delay)

    # Calculate summary stats
    duration = time.time() - start_time
    total_classifications = success_count + error_count
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    low_confidence_count = sum(1 for c in confidences if c < 0.7)

    # Log summary
    log_run_summary(
        logger,
        run_id,
        total_classifications,
        success_count,
        error_count,
        avg_confidence,
        low_confidence_count,
        duration,
    )

    # Update run record in database
    if db:
        db.complete_run(
            run_id=run_id,
            total_classifications=total_classifications,
            success_count=success_count,
            error_count=error_count,
            avg_confidence=avg_confidence,
            low_confidence_count=low_confidence_count,
            duration_seconds=duration,
        )

    # Export results
    if export_results and all_results:
        exporter = DataExporter()

        # Export for annotation
        annotation_files = exporter.export_for_annotation(run_id, all_results)
        logger.info(f"Exported annotation files to: {annotation_files.get('snippets_csv')}")

        # Export full results
        results_files = exporter.export_full_results(run_id, all_results, config)
        logger.info(f"Exported full results to: {results_files.get('full_json')}")

        # Export run metadata
        results_summary = {
            "total_classifications": total_classifications,
            "success_rate": success_count / total_classifications if total_classifications else 0,
            "avg_confidence": avg_confidence,
            "low_confidence_count": low_confidence_count,
        }
        metadata_path = export_run_metadata(run_id, config, results_summary)
        logger.info(f"Exported run metadata to: {metadata_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Total Classifications: {total_classifications}")
    print(f"Success: {success_count} ({100*success_count/max(total_classifications,1):.1f}%)")
    print(f"Errors: {error_count}")
    print(f"Average Confidence: {avg_confidence:.2f}")
    print(f"Low Confidence (<0.7): {low_confidence_count}")
    print("=" * 60)

    return run_id


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive classifier test suite"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2023, 2024],
        help="Years to process (default: 2023 2024)",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["harms", "adoption", "substantiveness"],
        choices=list(AVAILABLE_CLASSIFIERS.keys()),
        help="Classifiers to run (default: harms adoption substantiveness)",
    )
    parser.add_argument(
        "--companies-csv",
        type=Path,
        default=COMPANIES_CSV,
        help="Path to companies CSV file",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Model to use (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Don't save results to database",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Don't export annotation files",
    )

    args = parser.parse_args()

    # Run the test suite
    run_id = run_full_test_suite(
        companies_csv=args.companies_csv,
        years=args.years,
        classifiers=args.classifiers,
        model=args.model,
        rate_limit_delay=args.rate_limit,
        save_to_db=not args.no_db,
        export_results=not args.no_export,
    )

    print(f"\nRun completed: {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

