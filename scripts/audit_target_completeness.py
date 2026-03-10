#!/usr/bin/env python3
import csv
import json
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env.local", override=True)

API_KEY = os.environ.get("FR_API_KEY")
API_BASE = "https://api.financialreports.eu"
TARGET_MANIFEST = REPO_ROOT / "data" / "reference" / "target_manifest.csv"
OUTPUT_FILE = REPO_ROOT / "data" / "reference" / "audit_completeness.json"

TARGET_YEARS = [2021, 2022, 2023, 2024]

def get_filings_for_lei(lei, name):
    headers = {"x-api-key": API_KEY, "Accept": "application/json"}
    # Global search using company name + "Annual Report"
    params = {
        "search": f"{name} Annual Report",
        "page_size": 100,
    }
    try:
        resp = requests.get(f"{API_BASE}/filings/", headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            # Filter results for the correct LEI locally
            return [f for f in results if f.get("company", {}).get("lei") == lei]
        else:
            print(f"Error fetching {lei}: {resp.status_code}")
            return []
    except Exception as e:
        print(f"Request failed for {lei}: {e}")
        return []

def main():
    if not API_KEY:
        print("FR_API_KEY not found in .env.local")
        return

    # Load unique LEIs from target_manifest
    leis = {}
    with open(TARGET_MANIFEST, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            leis[row["lei"]] = row["company_name"]

    print(f"Auditing {len(leis)} companies...")
    
    results = {}
    
    count = 0
    import re
    for lei, name in leis.items():
        count += 1
        if count % 50 == 0:
            print(f"Processed {count}/{len(leis)}...")
        
        filings = get_filings_for_lei(lei, name)
        
        # Organize filings by fiscal year
        found_years = {}
        for f in filings:
            ftype = f.get("filing_type", {})
            fid = ftype.get("id")
            title = f.get("title", "")
            
            # Broaden filter since search is already targeted
            if fid not in [1, 2] and "Annual Report" not in title and "Annual Financial" not in title:
                continue
                
            # Try to get year from various fields
            year = f.get("fiscal_year")
            if not year:
                ped = f.get("period_end_date")
                if ped:
                    year = int(ped.split("-")[0])
            
            # If still no year, try parsing title (e.g. "Annual Report 2023")
            if not year:
                match = re.search(r"\b(202[0-5])\b", title)
                if match:
                    year = int(match.group(1))

            if year and int(year) in TARGET_YEARS:
                year_int = int(year)
                if year_int not in found_years:
                    found_years[year_int] = []
                found_years[year_int].append({
                    "pk": f.get("id"),
                    "status": f.get("processing_status"),
                    "is_esef": f.get("is_esef"),
                    "title": title
                })

        missing = [y for y in TARGET_YEARS if y not in found_years]
        
        results[lei] = {
            "name": name,
            "found_years": found_years,
            "missing_years": missing,
            "is_complete": len(missing) == 0
        }
        
        time.sleep(0.02) 

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Audit saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
