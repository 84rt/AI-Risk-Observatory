"""Main pipeline orchestrator for AIRO data processing."""

import csv
import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .aggregator import aggregate_firm
from .chunker import chunk_report
from .companies_house import fetch_reports_batch
from .config import get_settings
from .database import Database
from .llm_classifier import classify_candidates
from .pdf_extractor import extract_text_from_pdf
from .ixbrl_extractor import extract_text_from_ixbrl

logger = logging.getLogger(__name__)
settings = get_settings()


class Pipeline:
    """Main pipeline orchestrator."""

    def __init__(
        self,
        companies_csv: Path,
        year: Optional[int] = None,
        output_dir: Optional[Path] = None,
        clean_text: bool = False
    ):
        """Initialize the pipeline.

        Args:
            companies_csv: Path to CSV with company data
            year: Optional specific year to fetch. If not provided, gets latest.
            output_dir: Output directory for PDFs
            clean_text: Enable Gemini text cleaning (default: False, currently disabled due to summarization issues)
        """
        self.companies_csv = companies_csv
        self.year = year
        self.output_dir = output_dir or settings.raw_dir
        self.clean_text = clean_text
        self.db = Database()

        # Load companies
        self.companies = self._load_companies()
        logger.info(f"Loaded {len(self.companies)} companies")

    def _load_companies(self) -> List[dict]:
        """Load companies from CSV.

        Returns:
            List of company dicts
        """
        companies = []
        with open(self.companies_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                companies.append({
                    'ticker': row['ticker'],
                    'company_number': row['company_number'],
                    'company_name': row['company_name'],
                    'sector': row['sector']
                })
        return companies

    def run(self, skip_download: bool = False):
        """Run the complete pipeline.

        Args:
            skip_download: Skip download step if PDFs already exist
        """
        logger.info("=" * 80)
        logger.info("AIRO PIPELINE - Starting")
        logger.info("=" * 80)

        # Step 1: Download reports
        if not skip_download:
            logger.info("\n[1/5] Downloading annual reports from Companies House...")
            report_paths = self._download_reports()
        else:
            logger.info("\n[1/5] Skipping download (using existing reports)...")
            report_paths = self._find_existing_reports()

        # Step 2: Extract text and chunk
        logger.info("\n[2/5] Extracting text and chunking documents...")
        all_candidates = []
        for company in tqdm(self.companies, desc="Processing reports"):
            candidates = self._process_company(company, report_paths)
            if candidates:
                all_candidates.extend(candidates)

        logger.info(f"Generated {len(all_candidates)} candidate spans")

        # Step 3: LLM Classification
        logger.info("\n[3/5] Classifying spans with LLM...")
        results = classify_candidates(all_candidates)

        # Count relevant
        relevant_count = sum(1 for _, cls in results if cls.is_relevant)
        logger.info(
            f"Found {relevant_count} relevant mentions "
            f"out of {len(results)} candidates"
        )

        # Step 4: Save to database
        logger.info("\n[4/5] Saving mentions to database...")
        self.db.save_mentions_batch(
            results=results,
            model_version=settings.gemini_model
        )

        # Step 5: Aggregate to firm-level
        logger.info("\n[5/5] Aggregating to firm-level metrics...")
        for company in tqdm(self.companies, desc="Aggregating firms"):
            firm_id = company['ticker']
            year = self.year or 2024  # Default to 2024 if not specified

            try:
                aggregate_firm(firm_id, year, self.db)
            except Exception as e:
                logger.error(
                    f"Failed to aggregate {company['company_name']}: {e}"
                )

        logger.info("\n" + "=" * 80)
        logger.info("AIRO PIPELINE - Complete!")
        logger.info("=" * 80)

    def _download_reports(self) -> dict:
        """Download annual reports for all companies.

        Returns:
            Dict mapping company_number to result dict with 'path' and 'format' keys
        """
        report_results = fetch_reports_batch(
            companies=self.companies,
            year=self.year,
            output_dir=self.output_dir
        )
        return report_results

    def _find_existing_reports(self) -> dict:
        """Find existing reports (iXBRL or PDF) in output directory.

        Returns:
            Dict mapping company_number to result dict with 'path' and 'format' keys
        """
        ixbrl_dir = self.output_dir / "ixbrl"
        pdf_dir = self.output_dir / "pdfs"
        # Legacy structure support
        legacy_ixbrl_dir = self.output_dir / "reports" / "ixbrl"
        legacy_pdf_dir = self.output_dir / "reports" / "pdfs"
        
        report_paths = {}

        for company in self.companies:
            company_number = company['company_number']
            found = False
            
            # Prefer iXBRL if available
            matching_ixbrl = []
            if ixbrl_dir.exists():
                matching_ixbrl = list(ixbrl_dir.glob(f"{company_number}_*.xhtml"))
            if not matching_ixbrl and legacy_ixbrl_dir.exists():
                matching_ixbrl = list(legacy_ixbrl_dir.glob(f"{company_number}_*.xhtml"))
                if matching_ixbrl:
                    report_paths[company_number] = {
                        "path": matching_ixbrl[0],
                        "format": "ixbrl"
                    }
                    found = True
            
            # Fall back to PDF
            if not found:
                matching_pdfs = []
                if pdf_dir.exists():
                    matching_pdfs = list(pdf_dir.glob(f"{company_number}_*.pdf"))
                if not matching_pdfs and legacy_pdf_dir.exists():
                    matching_pdfs = list(legacy_pdf_dir.glob(f"{company_number}_*.pdf"))
                
                if matching_pdfs:
                    report_paths[company_number] = {
                        "path": matching_pdfs[0],
                        "format": "pdf"
                    }
                    found = True
            
            if not found:
                logger.warning(
                    f"No report found for {company['company_name']} "
                    f"({company_number})"
                )
                report_paths[company_number] = None

        return report_paths

    def _process_company(
        self,
        company: dict,
        report_paths: dict
    ) -> Optional[List]:
        """Process a single company's report.

        Args:
            company: Company dict
            report_paths: Dict mapping company_number to result dict with 'path' and 'format'

        Returns:
            List of CandidateSpan objects, or None if failed
        """
        company_number = company['company_number']
        report_info = report_paths.get(company_number)

        if not report_info or not report_info.get('path'):
            logger.warning(
                f"No report available for {company['company_name']}"
            )
            return None

        report_path = Path(report_info['path'])
        report_format = report_info.get('format', 'pdf')

        if not report_path.exists():
            logger.warning(
                f"Report file does not exist: {report_path} "
                f"for {company['company_name']}"
            )
            return None

        try:
            # Extract text based on format
            if report_format == 'ixbrl':
                logger.info(f"Extracting from iXBRL/XHTML: {company['company_name']}")
                extracted = extract_text_from_ixbrl(report_path)
            else:
                logger.info(f"Extracting from PDF: {company['company_name']}")
                extracted = extract_text_from_pdf(report_path)

            # Chunk
            candidates = chunk_report(
                extracted_report=extracted,
                firm_id=company['ticker'],
                firm_name=company['company_name'],
                sector=company['sector'],
                report_year=self.year or 2024
            )

            logger.info(
                f"Processed {company['company_name']} ({report_format}): "
                f"{len(candidates)} candidates"
            )

            return candidates

        except Exception as e:
            logger.error(
                f"Error processing {company['company_name']}: {e}",
                exc_info=True
            )
            return None


def run_pipeline(
    companies_csv: Path,
    year: Optional[int] = None,
    skip_download: bool = False,
    clean_text: bool = False
):
    """Convenience function to run the pipeline.

    Args:
        companies_csv: Path to companies CSV file
        year: Optional specific year to process
        skip_download: Skip download step if PDFs already exist
        clean_text: Enable Gemini text cleaning (default: False, currently disabled)
    """
    pipeline = Pipeline(
        companies_csv=companies_csv,
        year=year,
        clean_text=clean_text
    )
    pipeline.run(skip_download=skip_download)
