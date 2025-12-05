#!/usr/bin/env python3
"""Check document link structure."""

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

# Get document metadata
company_number = "00048839"
print(f"Checking document structure for {company_number}...\n")

# Get filing history
filing_url = f"https://api.company-information.service.gov.uk/company/{company_number}/filing-history"
response = requests.get(filing_url, headers=headers, params={"category": "accounts"})

if response.status_code == 200:
    items = response.json().get('items', [])
    if items:
        filing = items[0]
        doc_metadata_link = filing.get('links', {}).get('document_metadata')

        print(f"Document metadata link:")
        print(f"  {doc_metadata_link}\n")

        # Get metadata
        if doc_metadata_link:
            print(f"Fetching metadata...")
            metadata_response = requests.get(doc_metadata_link, headers=headers)

            if metadata_response.status_code == 200:
                metadata = metadata_response.json()
                print(f"\n Full metadata:")
                import json
                print(json.dumps(metadata, indent=2))

                content_link = metadata.get('links', {}).get('document')
                print(f"\nContent link: {content_link}")

                # Test if we need to add /content or not
                test_urls = [
                    content_link,
                    f"{content_link}/content",
                ]

                print(f"\nTesting URLs:")
                for url in test_urls:
                    print(f"\n  {url}")
                    test_response = requests.head(url, headers=headers, allow_redirects=True)
                    print(f"    Status: {test_response.status_code}")
                    print(f"    Content-Type: {test_response.headers.get('content-type')}")
            else:
                print(f"Failed to get metadata: {metadata_response.status_code}")
