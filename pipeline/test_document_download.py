#!/usr/bin/env python3
"""Test document download from Companies House."""

import sys
sys.path.insert(0, 'src')

import requests
from base64 import b64encode
from src.config import get_settings

settings = get_settings()
api_key = settings.companies_house_api_key

credentials = b64encode(f"{api_key}:".encode()).decode()
headers = {
    "Authorization": f"Basic {credentials}",
    "Accept": "application/json"
}

# Get filing history for Barclays
company_number = "01026167"
print(f"Getting filings for {company_number}...")

filing_url = f"https://api.company-information.service.gov.uk/company/{company_number}/filing-history"
response = requests.get(filing_url, headers=headers, params={"category": "accounts"})

if response.status_code == 200:
    data = response.json()
    items = data.get('items', [])
    print(f"Found {len(items)} filings\n")

    if items:
        first_filing = items[0]
        print("First filing:")
        print(f"  Description: {first_filing.get('description')}")
        print(f"  Date: {first_filing.get('date')}")
        print(f"  Type: {first_filing.get('type')}")

        # Check for document links
        links = first_filing.get('links', {})
        print(f"\n  Available links:")
        for key, value in links.items():
            print(f"    {key}: {value}")

        # Try to get document metadata
        doc_metadata_link = links.get('document_metadata')
        if doc_metadata_link:
            print(f"\nFetching document metadata...")
            # Note: Need to prepend base URL
            metadata_url = f"https://api.company-information.service.gov.uk{doc_metadata_link}"
            print(f"URL: {metadata_url}")

            metadata_response = requests.get(metadata_url, headers=headers)
            print(f"Status: {metadata_response.status_code}")

            if metadata_response.status_code == 200:
                metadata = metadata_response.json()
                print("\nDocument metadata:")
                for key, value in metadata.items():
                    if key == 'links':
                        print(f"  {key}:")
                        for link_key, link_value in value.items():
                            print(f"    {link_key}: {link_value}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"Failed: {metadata_response.text}")
else:
    print(f"Failed to get filings: {response.status_code}")
    print(response.text)
