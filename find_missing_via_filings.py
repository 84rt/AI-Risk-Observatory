import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[0]
load_dotenv(REPO_ROOT / ".env.local")

API_BASE = "https://api.financialreports.eu"
API_KEY = os.environ.get("FR_API_KEY", "")
HEADERS = {"x-api-key": API_KEY, "Accept": "application/json"}

def find_company_from_filing(query):
    print(f"\n--- Searching Filings for '{query}' ---")
    r = requests.get(f"{API_BASE}/filings/", params={"search": query, "page_size": 20}, headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        print(f"Filings found: {data.get('count', 0)}")
        found_ids = {}
        for res in data.get('results', []):
            co = res.get('company', {})
            if co.get('id'):
                found_ids[co.get('id')] = co.get('name')
        
        for cid, cname in found_ids.items():
            print(f"  MATCH: Company ID: {cid} | Name: {cname}")
    else:
        print(f"Error: {r.status_code} - {r.text}")

# Specific targets from the unmatched list
find_company_from_filing("Hiscox")
find_company_from_filing("Experian")
find_company_from_filing("Entain")
find_company_from_filing("DCC")
find_company_from_filing("Coca-Cola HBC")
find_company_from_filing("International Airlines")
find_company_from_filing("Spirax")
find_company_from_filing("B&M")
find_company_from_filing("Pershing Square")
find_company_from_filing("Phoenix Group")
find_company_from_filing("Howdens")
find_company_from_filing("Intertek")
