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

def search_global(name):
    print(f"\n--- Global API Search for '{name}' ---")
    r = requests.get(f"{API_BASE}/companies/", params={"search": name}, headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        print(f"Results: {data.get('count', 0)}")
        for res in data.get('results', []):
            print(f"  ID: {res['id']} | Name: {res['name']} | LEI: {res.get('lei')}")
    else:
        print(f"  Error: {r.status_code}")

targets = ["Experian", "Entain", "Hiscox", "International Consolidated Airlines", "Coca-Cola Europacific"]
for t in targets:
    search_global(t)
