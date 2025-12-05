#!/usr/bin/env python3
"""Quick test to inspect document metadata structure."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.companies_house import CompaniesHouseClient
import json

client = CompaniesHouseClient()

# Get filing
filing = client.get_latest_annual_accounts("04366849", year=2024)
doc_metadata_link = filing.get("links", {}).get("document_metadata")

print("Document metadata link:")
print(f"  {doc_metadata_link}\n")

# Get metadata
metadata = client._get_document_metadata(doc_metadata_link)

print("Full metadata:")
print(json.dumps(metadata, indent=2))
