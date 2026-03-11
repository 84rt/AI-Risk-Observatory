import csv
import os
import sys
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[0]
load_dotenv(REPO_ROOT / ".env.local")

API_BASE = "https://api.financialreports.eu"
API_KEY = os.environ.get("FR_API_KEY", "")
HEADERS = {"x-api-key": API_KEY, "Accept": "application/json"}

def normalize_ticker(t):
    if not t: return ""
    return t.split("-")[0].split(".")[0].strip().upper()

def normalize_name(n):
    if not n: return ""
    # Standard normalization for comparison
    n = n.lower().replace(" plc", "").replace(" ltd", "").replace(" holdings", "").replace(" group", "").replace(" investment trusts", "").replace(" investment trust", "").strip()
    return "".join(filter(str.isalnum, n))

# 1. Load 350 constituents
ftse_companies = []
with open("ftse350_constituents.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['norm_ticker'] = normalize_ticker(row['Ticker'])
        row['norm_name'] = normalize_name(row['Company'])
        ftse_companies.append(row)

# 2. Load local FR dataset
fr_local = {}
fr_by_name = {}
local_path = Path("data/FR_dataset/companies.csv")
if local_path.exists():
    with open(local_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = normalize_ticker(row['ticker'])
            if t: fr_local[t] = row
            n = normalize_name(row['name'])
            if n: fr_by_name[n] = row

# 3. Match
found = []
missing = []

for co in ftse_companies:
    match = fr_local.get(co['norm_ticker']) or fr_by_name.get(co['norm_name'])
    if match:
        co['fr_id'] = match['id']
        found.append(co)
    else:
        missing.append(co)

print(f"Locally found: {len(found)}")
print(f"Locally missing: {len(missing)}")

# 4. API check for missing
if missing and API_KEY:
    print(f"Checking API for {len(missing)} missing companies...")
    for co in missing:
        ticker = co['norm_ticker']
        try:
            r = requests.get(f"{API_BASE}/companies/", params={"ticker": ticker, "listed_stock_exchange": 10}, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    co['fr_id'] = results[0]['id']
                    found.append(co)
                    print(f"  ✅ Found via Ticker: {co['Company']} ({ticker})")
                    time.sleep(0.1)
                    continue
            
            # Try name search if ticker failed
            r = requests.get(f"{API_BASE}/companies/", params={"search": co['Company']}, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    co['fr_id'] = results[0]['id']
                    found.append(co)
                    print(f"  ✅ Found via Search: {co['Company']}")
                else:
                    print(f"  ❓ Not found: {co['Company']}")
            
            time.sleep(0.1)
        except Exception as e:
            print(f"  ❌ API Error for {co['Company']}: {e}")

print(f"\nFinal Summary:")
print(f"Total Companies in list: {len(ftse_companies)}")
print(f"Matched in FR Database: {len(found)}")
print(f"Missing from FR: {len(ftse_companies) - len(found)}")
