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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def download_document(
        self,
        document_metadata_link: str,
        output_path: Path
    ) -> Path:
        """Download a document from Companies House.

        Args:
            document_metadata_link: The document metadata link from filing history
            output_path: Where to save the PDF

        Returns:
            Path to downloaded file

        Raises:
            requests.HTTPError: If the download fails
        """
        # Get document metadata to find content link
        logger.info(f"Fetching document metadata from {document_metadata_link}")
        # Check if link is already a full URL
        if document_metadata_link.startswith('http'):
            metadata_url = document_metadata_link
        else:
            metadata_url = f"{self.base_url}{document_metadata_link}"

        metadata_response = self.session.get(metadata_url)
        metadata_response.raise_for_status()

        metadata = metadata_response.json()
        content_link = metadata.get("links", {}).get("document")

        if not content_link:
            raise ValueError(f"No content link found in metadata: {metadata}")

        # Download the actual document
        # The content_link from metadata is already the full URL with /content
        # If it's already a full URL, use it directly; otherwise construct it
        if content_link.startswith('http'):
            content_url = content_link  # Already includes /content
        else:
            # If relative, append /content
            content_url = f"{self.base_url}{content_link}/content"
        logger.info(f"Downloading document from {content_url}")

        # Use stream=True for large files
        response = self.session.get(
            content_url,
            headers={"Accept": "application/pdf"},
            stream=True
        )
        response.raise_for_status()

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Document downloaded to {output_path}")
        return output_path

    def fetch_annual_report(
        self,
        company_number: str,
        company_name: str,
        year: Optional[int] = None,
        output_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Fetch the annual report for a company.

        Args:
            company_number: The company number
            company_name: The company name (for filename)
            year: Optional specific year. If not provided, gets the latest.
            output_dir: Where to save PDFs. If not provided, uses settings.

        Returns:
            Path to downloaded PDF or None if not found
        """
        if output_dir is None:
            output_dir = settings.output_dir / "pdfs"

        # Get the filing
        filing = self.get_latest_annual_accounts(company_number, year)
        if not filing:
            return None

        # Get document metadata link
        doc_metadata_link = filing.get("links", {}).get("document_metadata")
        if not doc_metadata_link:
            logger.warning(
                f"No document metadata link for company {company_number}"
            )
            return None

        # Generate filename
        filing_year = self._extract_year(filing) or "unknown"
        safe_name = company_name.replace(" ", "_").replace("/", "_")
        filename = f"{company_number}_{safe_name}_{filing_year}.pdf"
        output_path = output_dir / filename

        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"Document already exists: {output_path}")
            return output_path

        # Download
        try:
            return self.download_document(doc_metadata_link, output_path)
        except Exception as e:
            logger.error(
                f"Failed to download document for {company_number}: {e}"
            )
            return None


def fetch_reports_batch(
    companies: List[Dict[str, str]],
    year: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Optional[Path]]:
    """Fetch annual reports for a batch of companies.

    Args:
        companies: List of dicts with keys: company_number, company_name
        year: Optional specific year
        output_dir: Where to save PDFs

    Returns:
        Dict mapping company_number to Path (or None if failed)
    """
    client = CompaniesHouseClient()
    results = {}

    for company in companies:
        company_number = company["company_number"]
        company_name = company["company_name"]

        logger.info(f"Fetching report for {company_name} ({company_number})")

        try:
            path = client.fetch_annual_report(
                company_number=company_number,
                company_name=company_name,
                year=year,
                output_dir=output_dir
            )
            results[company_number] = path
        except Exception as e:
            logger.error(f"Error fetching report for {company_number}: {e}")
            results[company_number] = None

    return results
