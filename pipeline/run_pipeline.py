#!/usr/bin/env python3
"""Command-line interface for running the AIRO pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add pipeline directory to path so we can import src as a package
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_pipeline
from src.config import get_settings

settings = get_settings()
DEFAULT_COMPANIES = settings.data_dir / "reference" / "companies_template.csv"
LOG_DIR = settings.logs_dir / "pipeline"


def setup_logging(level: str = "INFO"):
    """Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "pipeline.log")
        ]
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AIRO Data Pipeline - Process UK company annual reports for AI risk mentions"
    )

    parser.add_argument(
        "--companies",
        type=Path,
        default=DEFAULT_COMPANIES,
        help=f"Path to companies CSV file (default: {DEFAULT_COMPANIES})"
    )

    parser.add_argument(
        "--year",
        type=int,
        help="Specific report year to fetch (e.g., 2024). If not provided, fetches latest."
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step and use existing PDFs"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--clean-text",
        action="store_true",
        help="Enable Gemini text cleaning (EXPERIMENTAL: currently disabled due to summarization issues)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)

    # Validate companies file
    if not args.companies.exists():
        logger.error(f"Companies file not found: {args.companies}")
        logger.error(
            "Please create the file or use the template at "
            f"{DEFAULT_COMPANIES}"
        )
        sys.exit(1)

    # Check API keys
    try:
        _ = settings.gemini_api_key
        _ = settings.companies_house_api_key
    except Exception as e:
        logger.error(
            "Missing API keys! Please create a .env or .env.local file with:\n"
            "  GEMINI_API_KEY=your_key_here\n"
            "  COMPANIES_HOUSE_API_KEY=your_key_here"
        )
        sys.exit(1)

    # Run pipeline
    try:
        logger.info("Starting AIRO pipeline...")
        clean_text = args.clean_text
        if clean_text:
            logger.info("Text cleaning with Gemini 2.0 Flash: ENABLED (EXPERIMENTAL)")
            logger.warning("Note: Gemini cleaning currently disabled due to summarization issues")
        else:
            logger.info("Text cleaning with Gemini 2.0 Flash: DISABLED (using regex-based cleanup)")

        run_pipeline(
            companies_csv=args.companies,
            year=args.year,
            skip_download=args.skip_download,
            clean_text=clean_text
        )
        logger.info("Pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
