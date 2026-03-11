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

def find_via_filing(name):
    print(f"\n--- Finding Company for '{name}' via filings ---")
    r = requests.get(f"{API_BASE}/filings/", params={"search": name, "page_size": 1}, headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        if data['results']:
            co = data['results'][0]['company']
            print(f"  MATCH: {co['name']} | ID: {co['id']}")
        else:
            print("  No filings found.")
    else:
        print(f"  Error: {r.status_code}")

targets = ["Experian", "Entain", "Hiscox", "International Airlines", "Coca-Cola HBC", "DCC plc", "B&M European", "Pershing Square", "Applied Nutrition", "Ashtead"]
for t in targets:
    find_via_filing(t)
