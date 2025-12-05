"""Companies House API client for fetching annual reports."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from base64 import b64encode

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CompaniesHouseClient:
    """Client for interacting with Companies House API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client with API credentials.

        Args:
            api_key: Companies House API key. If not provided, uses settings.
        """
        self.api_key = api_key or settings.companies_house_api_key
        self.base_url = settings.companies_house_base_url
        self.session = requests.Session()

        # Companies House uses HTTP Basic Auth with API key as username
        credentials = b64encode(f"{self.api_key}:".encode()).decode()
        self.session.headers.update({
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json"
        })

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def get_filing_history(
        self,
        company_number: str,
        category: str = "accounts"
    ) -> List[Dict]:
        """Get filing history for a company.

        Args:
            company_number: The company number (e.g., "00489800")
            category: Filing category (default: "accounts")

        Returns:
            List of filing records

        Raises:
            requests.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/company/{company_number}/filing-history"
        params = {"category": category}

        logger.info(f"Fetching filing history for company {company_number}")
        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get("items", [])

    def get_latest_annual_accounts(
        self,
        company_number: str,
        year: Optional[int] = None
    ) -> Optional[Dict]:
        """Get the most recent annual accounts filing.

        Args:
            company_number: The company number
            year: Optional specific year to fetch. If not provided, gets the latest.

        Returns:
            Filing record dict or None if not found
        """
        filings = self.get_filing_history(company_number, category="accounts")

        # Filter for Annual Accounts (AA type)
        annual_accounts = [
            f for f in filings
            if f.get("type") == "AA" or "annual" in f.get("description", "").lower()
        ]

        if not annual_accounts:
            logger.warning(f"No annual accounts found for company {company_number}")
            return None

        # If year specified, filter by year
        if year:
            annual_accounts = [
                f for f in annual_accounts
                if self._extract_year(f) == year
            ]

        if not annual_accounts:
            logger.warning(
                f"No annual accounts found for company {company_number} "
                f"in year {year}"
            )
            return None

        # Return the most recent
        return annual_accounts[0]

    @staticmethod
    def _extract_year(filing: Dict) -> Optional[int]:
        """Extract year from filing record.

        Args:
            filing: Filing record dict

        Returns:
            Year as integer or None
        """
        date_str = filing.get("date")
        if date_str:
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                return date_obj.year
            except ValueError:
                pass

        # Try to extract from description
        description = filing.get("description", "")
        for token in description.split():
            if token.isdigit() and len(token) == 4:
                year = int(token)
                if 2000 <= year <= 2030:
                    return year

        return None

    def _get_document_metadata(self, metadata_link: str) -> Dict:
        """Get document metadata to check available formats.

        Args:
            metadata_link: The document metadata link from filing history

        Returns:
            Metadata dict with 'resources' showing available formats
        """
        logger.debug(f"Fetching document metadata from {metadata_link}")

        if metadata_link.startswith('http'):
            metadata_url = metadata_link
        else:
            metadata_url = f"{self.base_url}{metadata_link}"

        metadata_response = self.session.get(metadata_url)
        metadata_response.raise_for_status()

        return metadata_response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def download_ixbrl(
        self,
        ixbrl_link: str,
        output_path: Path
    ) -> Path:
        """Download an iXBRL/XHTML document from Companies House.

        This handles the redirect to Amazon S3 properly to avoid auth errors.

        Args:
            ixbrl_link: The iXBRL link from filing history
            output_path: Where to save the file

        Returns:
            Path to downloaded file

        Raises:
            requests.HTTPError: If the download fails
        """
        logger.info(f"Downloading iXBRL/XHTML from {ixbrl_link}")

        # iXBRL links are typically direct URLs, but check if relative
        if ixbrl_link.startswith('http'):
            content_url = ixbrl_link
        else:
            content_url = f"{self.base_url}{ixbrl_link}"

        # CRITICAL: Use allow_redirects=False to prevent sending API key to S3
        # Request XHTML format explicitly
        response = self.session.get(
            content_url,
            headers={"Accept": "application/xhtml+xml"},
            allow_redirects=False,
            stream=True
        )

        # Handle redirect manually
        final_url = None
        if response.status_code == 302 or response.status_code == 301:
            # Got redirect to S3 - grab the pre-signed URL
            final_url = response.headers.get('Location')
            logger.debug(f"Following redirect to S3: {final_url[:80]}...")
        elif response.status_code == 200:
            # Sometimes returns directly (rare)
            final_url = content_url
        else:
            response.raise_for_status()

        # Download from S3 without auth headers (URL is pre-signed)
        if final_url != content_url:
            logger.debug("Downloading from S3 (no auth headers)...")
            file_response = requests.get(final_url, stream=True)
        else:
            file_response = response

        file_response.raise_for_status()

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"iXBRL/XHTML downloaded to {output_path}")
        return output_path

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def download_document(
        self,
        document_metadata_link: str,
        output_path: Path,
        prefer_xhtml: bool = True
    ) -> Path:
        """Download a document from Companies House (XHTML preferred, PDF fallback).

        This method checks metadata for available formats and handles S3 redirects properly.

        Args:
            document_metadata_link: The document metadata link from filing history
            output_path: Where to save the file
            prefer_xhtml: If True, tries to get XHTML/iXBRL format first

        Returns:
            Path to downloaded file

        Raises:
            requests.HTTPError: If the download fails
        """
        # Get document metadata to check available formats
        metadata = self._get_document_metadata(document_metadata_link)

        # Check what formats are available in resources
        resources = metadata.get("resources", {})
        logger.info(f"Available formats in metadata: {list(resources.keys())}")

        # Determine which format to download (priority: XHTML -> ZIP -> PDF)
        target_format = None
        file_extension = None

        if prefer_xhtml and "application/xhtml+xml" in resources:
            target_format = "application/xhtml+xml"
            file_extension = ".xhtml"
            logger.info("Found XHTML format (iXBRL), downloading...")
        elif prefer_xhtml and "application/zip" in resources:
            target_format = "application/zip"
            file_extension = ".zip"
            logger.info("Found ZIP format (ESEF package), downloading...")
        elif "application/pdf" in resources or not resources:
            # Default to PDF if no resources listed or PDF explicitly available
            target_format = "application/pdf"
            file_extension = ".pdf"
            logger.info("Downloading PDF format...")
        else:
            raise ValueError(f"No downloadable format found. Available: {list(resources.keys())}")

        # Update output path extension if needed
        if not str(output_path).endswith(file_extension):
            output_path = output_path.with_suffix(file_extension)

        # Get content link
        content_link = metadata.get("links", {}).get("document")
        if not content_link:
            raise ValueError(f"No content link found in metadata: {metadata}")

        # Build content URL
        if content_link.startswith('http'):
            content_url = f"{content_link}/content"
        else:
            content_url = f"{self.base_url}{content_link}/content"

        logger.info(f"Requesting {target_format} from {content_url}")

        # CRITICAL: Use allow_redirects=False to prevent sending API key to S3
        response = self.session.get(
            content_url,
            headers={"Accept": target_format},
            allow_redirects=False,
            stream=True
        )

        # Handle redirect manually
        final_url = None
        if response.status_code == 302 or response.status_code == 301:
            # Got redirect to S3 - grab the pre-signed URL
            final_url = response.headers.get('Location')
            logger.debug(f"Following redirect to S3: {final_url[:80]}...")
        elif response.status_code == 200:
            # Sometimes returns directly (rare for large files)
            final_url = content_url
        else:
            response.raise_for_status()

        # Download from S3 without auth headers (URL is pre-signed)
        if final_url != content_url:
            logger.debug("Downloading from S3 (no auth headers)...")
            file_response = requests.get(final_url, stream=True)
        else:
            file_response = response

        file_response.raise_for_status()

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Document downloaded to {output_path}")
        return output_path

    def fetch_annual_report(
        self,
        company_number: str,
        company_name: str,
        year: Optional[int] = None,
        output_dir: Optional[Path] = None
    ) -> Optional[Dict[str, any]]:
        """Fetch the annual report for a company.

        Prefers iXBRL/XHTML format, falls back to PDF if not available.

        Args:
            company_number: The company number
            company_name: The company name (for filename)
            year: Optional specific year. If not provided, gets the latest.
            output_dir: Where to save documents. If not provided, uses settings.

        Returns:
            Dict with keys: 'path' (Path), 'format' ('ixbrl' or 'pdf'), or None if not found
        """
        if output_dir is None:
            output_dir = settings.output_dir / "reports"

        # Get the filing
        filing = self.get_latest_annual_accounts(company_number, year)
        if not filing:
            return None

        links = filing.get("links", {})
        filing_year = self._extract_year(filing) or "unknown"
        safe_name = company_name.replace(" ", "_").replace("/", "_")

        # Try iXBRL direct link first (if available)
        # Companies House API may use either "ixbrl" or "xbrl" key
        ixbrl_link = links.get("ixbrl") or links.get("xbrl")
        if ixbrl_link:
            logger.info(f"Found iXBRL/XBRL direct link for {company_name}, attempting download...")
            filename = f"{company_number}_{safe_name}_{filing_year}.xhtml"
            output_path = output_dir / "ixbrl" / filename

            # Skip if already downloaded
            if output_path.exists():
                logger.info(f"iXBRL document already exists: {output_path}")
                return {"path": output_path, "format": "ixbrl"}

            try:
                downloaded_path = self.download_ixbrl(ixbrl_link, output_path)
                logger.info(f"✅ Downloaded iXBRL/XHTML for {company_name}")
                return {"path": downloaded_path, "format": "ixbrl"}
            except Exception as e:
                logger.warning(
                    f"Failed to download iXBRL from direct link for {company_number}: {e}"
                )
                # Fall through to document metadata download

        # Use document metadata endpoint (can provide XHTML, ZIP, or PDF)
        doc_metadata_link = links.get("document_metadata")
        if not doc_metadata_link:
            logger.warning(
                f"No document metadata link for company {company_number}"
            )
            return None

        # Generate base filename (extension will be determined by download method)
        base_filename = f"{company_number}_{safe_name}_{filing_year}"

        # Check if we already have this document in any format
        for check_dir, check_ext, check_format in [
            ("ixbrl", ".xhtml", "ixbrl"),
            ("ixbrl", ".zip", "ixbrl"),
            ("pdfs", ".pdf", "pdf")
        ]:
            check_path = output_dir / check_dir / f"{base_filename}{check_ext}"
            if check_path.exists():
                logger.info(f"Document already exists: {check_path}")
                return {"path": check_path, "format": check_format}

        # Download, preferring XHTML format
        output_path = output_dir / "ixbrl" / f"{base_filename}.xhtml"  # Initial path, may change based on format
        try:
            downloaded_path = self.download_document(
                doc_metadata_link,
                output_path,
                prefer_xhtml=True
            )

            # Determine format from file extension
            file_ext = downloaded_path.suffix.lower()
            if file_ext in ['.xhtml', '.html', '.zip']:
                format_type = "ixbrl"
            else:
                format_type = "pdf"

            logger.info(f"✅ Downloaded {format_type.upper()} for {company_name}")
            return {"path": downloaded_path, "format": format_type}
        except Exception as e:
            logger.error(
                f"Failed to download document for {company_number}: {e}"
            )
            return None


def fetch_reports_batch(
    companies: List[Dict[str, str]],
    year: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Optional[Dict[str, any]]]:
    """Fetch annual reports for a batch of companies.

    Args:
        companies: List of dicts with keys: company_number, company_name
        year: Optional specific year
        output_dir: Where to save documents

    Returns:
        Dict mapping company_number to result dict with 'path' and 'format' keys (or None if failed)
    """
    client = CompaniesHouseClient()
    results = {}

    for company in companies:
        company_number = company["company_number"]
        company_name = company["company_name"]

        logger.info(f"Fetching report for {company_name} ({company_number})")

        try:
            result = client.fetch_annual_report(
                company_number=company_number,
                company_name=company_name,
                year=year,
                output_dir=output_dir
            )
            results[company_number] = result
        except Exception as e:
            logger.error(f"Error fetching report for {company_number}: {e}")
            results[company_number] = None

    return results
