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

def deep_search(name):
    print(f"\n--- Searching for '{name}' ---")
    url = f"{API_BASE}/companies/"
    params = {"search": name, "page_size": 100}
    r = requests.get(url, params=params, headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        print(f"Results: {data.get('count')}")
        for res in data.get('results', []):
            print(f"  ID: {res.get('id')} | Name: {res.get('name')} | Ticker: {res.get('ticker')}")
    else:
        print(f"  Error: {r.status_code}")

# Targets
targets = ["Experian", "Hiscox", "Entain", "Coca-Cola", "DCC", "RS Group", "International Airlines", "Spirax", "B&M"]
for t in targets:
    deep_search(t)
