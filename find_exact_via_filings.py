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

# Exact names from the URLs provided
EXACT_NAMES = [
    "Hiscox Limited",
    "Experian PLC",
    "Entain PLC",
    "International Consolidated Airlines Group S.A.",
    "Coca-Cola Europacific Partners PLC"
]

def find_exact_via_filing(name):
    print(f"\n--- Finding Company for '{name}' via filings ---")
    r = requests.get(f"{API_BASE}/filings/", params={"search": name, "page_size": 1}, headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        if data['results']:
            co = data['results'][0]['company']
            print(f"  MATCH: {co['name']} | ID: {co['id']} | LEI: {co.get('lei')}")
        else:
            print("  No filings found.")
    else:
        print(f"  Error: {r.status_code}")

for name in EXACT_NAMES:
    find_exact_via_filing(name)
