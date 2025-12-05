"""Client for filings.xbrl.org API to download iXBRL reports."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


class XBRLFilingsClient:
    """Client for interacting with filings.xbrl.org API."""

    def __init__(self):
        """Initialize the client."""
        self.base_url = "https://filings.xbrl.org"
        self.api_url = f"{self.base_url}/api"

    def search_entity_by_lei(self, lei: str) -> Optional[Dict]:
        """Search for an entity by LEI code.

        Args:
            lei: Legal Entity Identifier

        Returns:
            Entity data dict or None if not found
        """
        url = f"{self.api_url}/entities/{lei}"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json().get("data")
        elif response.status_code == 404:
            logger.warning(f"Entity not found for LEI: {lei}")
            return None
        else:
            response.raise_for_status()

    def get_entity_filings(
        self,
        lei: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get filings for an entity by LEI.

        Args:
            lei: Legal Entity Identifier
            limit: Maximum number of filings to return

        Returns:
            List of filing records
        """
        params = {
            "page[size]": limit
        }

        # Get entity's filings link
        entity = self.search_entity_by_lei(lei)
        if not entity:
            return []

        filings_link = entity.get("relationships", {}).get("filings", {}).get("links", {}).get("related")
        if not filings_link:
            logger.warning(f"No filings link for entity: {lei}")
            return []

        url = f"{self.base_url}{filings_link}"
        response = requests.get(url, params=params)
        response.raise_for_status()

        return response.json().get("data", [])

    def get_latest_filing(self, lei: str) -> Optional[Dict]:
        """Get the most recent filing for an entity.

        Args:
            lei: Legal Entity Identifier

        Returns:
            Latest filing record or None
        """
        filings = self.get_entity_filings(lei, limit=1)
        return filings[0] if filings else None

    def download_xhtml_report(
        self,
        filing: Dict,
        output_path: Path
    ) -> Path:
        """Download XHTML report from a filing.

        Args:
            filing: Filing record from API
            output_path: Where to save the XHTML file

        Returns:
            Path to downloaded file

        Raises:
            ValueError: If filing doesn't have XHTML report
            requests.HTTPError: If download fails
        """
        attrs = filing.get("attributes", {})
        report_url = attrs.get("report_url")

        if not report_url:
            raise ValueError(f"Filing has no report_url: {filing.get('id')}")

        # Construct full URL
        if not report_url.startswith("http"):
            report_url = f"{self.base_url}{report_url}"

        logger.info(f"Downloading XHTML report from {report_url}")

        response = requests.get(report_url, stream=True)
        response.raise_for_status()

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"XHTML report downloaded to {output_path}")
        return output_path

    def download_package(
        self,
        filing: Dict,
        output_path: Path
    ) -> Path:
        """Download full ESEF package (ZIP) from a filing.

        Args:
            filing: Filing record from API
            output_path: Where to save the ZIP file

        Returns:
            Path to downloaded file

        Raises:
            ValueError: If filing doesn't have package
            requests.HTTPError: If download fails
        """
        attrs = filing.get("attributes", {})
        package_url = attrs.get("package_url")

        if not package_url:
            raise ValueError(f"Filing has no package_url: {filing.get('id')}")

        # Construct full URL
        if not package_url.startswith("http"):
            package_url = f"{self.base_url}{package_url}"

        logger.info(f"Downloading ESEF package from {package_url}")

        response = requests.get(package_url, stream=True)
        response.raise_for_status()

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"ESEF package downloaded to {output_path}")
        return output_path

    def fetch_annual_report(
        self,
        lei: str,
        entity_name: str,
        output_dir: Optional[Path] = None,
        year: Optional[int] = None
    ) -> Optional[Dict[str, any]]:
        """Fetch annual report for a company by LEI.

        Args:
            lei: Legal Entity Identifier
            entity_name: Company name (for filename)
            output_dir: Where to save the report
            year: Optional specific year (filters by period_end)

        Returns:
            Dict with 'path' and 'format' keys, or None if not found
        """
        if output_dir is None:
            output_dir = Path("output/reports/ixbrl")

        # Get filings
        filings = self.get_entity_filings(lei, limit=20)

        if not filings:
            logger.warning(f"No filings found for LEI: {lei}")
            return None

        # Filter by year if specified
        if year:
            filings = [
                f for f in filings
                if f.get("attributes", {}).get("period_end", "").startswith(str(year))
            ]

        if not filings:
            logger.warning(f"No filings found for LEI {lei} in year {year}")
            return None

        # Get the latest filing
        filing = filings[0]
        attrs = filing.get("attributes", {})
        period_end = attrs.get("period_end", "unknown")

        # Generate filename
        safe_name = entity_name.replace(" ", "_").replace("/", "_")
        filename = f"{lei}_{safe_name}_{period_end}.xhtml"
        output_path = output_dir / filename

        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"XHTML report already exists: {output_path}")
            return {"path": output_path, "format": "ixbrl"}

        # Download
        try:
            downloaded_path = self.download_xhtml_report(filing, output_path)
            return {"path": downloaded_path, "format": "ixbrl"}
        except Exception as e:
            logger.error(f"Failed to download report for {lei}: {e}")
            return None


def fetch_reports_batch_xbrl(
    companies: List[Dict[str, str]],
    year: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Optional[Dict[str, any]]]:
    """Fetch annual reports for a batch of companies using XBRL API.

    Args:
        companies: List of dicts with keys: lei, company_name
        year: Optional specific year
        output_dir: Where to save reports

    Returns:
        Dict mapping LEI to result dict with 'path' and 'format' keys (or None if failed)
    """
    client = XBRLFilingsClient()
    results = {}

    for company in companies:
        lei = company["lei"]
        company_name = company["company_name"]

        logger.info(f"Fetching report for {company_name} (LEI: {lei})")

        try:
            result = client.fetch_annual_report(
                lei=lei,
                entity_name=company_name,
                year=year,
                output_dir=output_dir
            )
            results[lei] = result
        except Exception as e:
            logger.error(f"Error fetching report for {lei}: {e}")
            results[lei] = None

    return results
