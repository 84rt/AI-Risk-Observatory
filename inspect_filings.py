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

def inspect_filings(name):
    print(f"\n--- Inspecting Filings for '{name}' ---")
    r = requests.get(f"{API_BASE}/filings/", params={"search": name, "page_size": 1}, headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        if data['results']:
            print(json.dumps(data['results'][0], indent=2))
        else:
            print("No filings found.")
    else:
        print(f"Error: {r.status_code}")

inspect_filings("Experian")
inspect_filings("Hiscox")
inspect_filings("Entain")
