import os
import csv
import json
import re
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env.local")

API_BASE = "https://api.financialreports.eu/api"
API_KEY = os.environ.get("FR_API_KEY", "")
HEADERS = {"x-api-key": API_KEY, "Accept": "application/json"}

TARGET_MANIFEST_PATH = REPO_ROOT / "data" / "reference" / "target_manifest.csv"
LSE_UNIVERSE_PATH = REPO_ROOT / "data" / "reference" / "lse_company_reports_universe.csv"
FR_MANIFEST_PATH = REPO_ROOT / "data" / "FR_dataset" / "manifest.csv"
OUTPUT_MAP_PATH = REPO_ROOT / "data" / "reference" / "fr_reconciliation_map.csv"

def normalize_name(name):
    if not name: return ""
    name = name.lower()
    # Remove common corporate suffixes and shares identifiers
    name = re.sub(r'\b(plc|ltd|limited|group|holdings|inc|corp|ord|npv|sa|nv)\b', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def load_fr_leis():
    fr_leis = set()
    if FR_MANIFEST_PATH.exists():
        with open(FR_MANIFEST_PATH, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('company__lei'):
                    fr_leis.add(row['company__lei'].strip().upper())
    return fr_leis

def load_lse_data():
    lse_map = {}
    if LSE_UNIVERSE_PATH.exists():
        with open(LSE_UNIVERSE_PATH, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                name = row.get('instrument_name', '')
                norm = normalize_name(name.split(' ORD ')[0])
                if norm:
                    lse_map[norm] = {
                        'isin': row.get('isin', '').strip().upper(),
                        'ticker': row.get('epic', '').strip().upper()
                    }
    return lse_map

def find_lse_identifiers(target_name, lse_map):
    norm_target = normalize_name(target_name)
    if not norm_target: return None, None
    
    # Exact normalized match
    if norm_target in lse_map:
        return lse_map[norm_target]['isin'], lse_map[norm_target]['ticker']
        
    # Substring match
    for lse_name, data in lse_map.items():
        if len(norm_target) > 3 and len(lse_name) > 3:
            if lse_name in norm_target or norm_target in lse_name:
                return data['isin'], data['ticker']
    
    return None, None

def query_fr_api(params):
    try:
        time.sleep(0.3) # Rate limit
        r = requests.get(f"{API_BASE}/companies/", headers=HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get('count', 0) > 0:
                return data['results'][0]
    except Exception as e:
        print(f"API Error with params {params}: {e}")
    return None

def main():
    if not API_KEY:
        print("ERROR: FR_API_KEY not set.")
        return

    print("Loading existing FR LEIs...")
    fr_leis = load_fr_leis()
    
    print("Loading LSE Universe data...")
    lse_map = load_lse_data()

    print("Finding missing companies from target manifest...")
    missing_companies = {}
    with open(TARGET_MANIFEST_PATH, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['fr_status'] == 'not_in_fr':
                lei = row.get('lei', '').strip().upper()
                name = row.get('company_name', '').strip()
                if lei and lei not in fr_leis:
                    if lei not in missing_companies:
                        missing_companies[lei] = name

    print(f"Found {len(missing_companies)} missing companies to reconcile.")
    
    results = []
    
    for i, (lei, name) in enumerate(missing_companies.items(), 1):
        print(f"[{i}/{len(missing_companies)}] Reconciling: {name} ({lei})")
        
        isin, ticker = find_lse_identifiers(name, lse_map)
        
        match = None
        match_method = None
        
        # 1. Try ISIN
        if isin and not match:
            print(f"  -> Trying ISIN: {isin}")
            match = query_fr_api({"isin": isin})
            if match: match_method = "ISIN"
            
        # 2. Try Ticker
        if ticker and not match:
            print(f"  -> Trying Ticker: {ticker}")
            match = query_fr_api({"ticker": ticker})
            if match: match_method = "Ticker"
            
        # 3. Try Name Search
        if not match:
            print(f"  -> Trying Name Search: {name}")
            # Use original name first, then normalized if it fails? The API search might be better with original.
            match = query_fr_api({"search": name})
            if match:
                # Basic sanity check on name match
                fr_name = normalize_name(match.get('name', ''))
                target_norm = normalize_name(name)
                if len(fr_name) > 3 and len(target_norm) > 3:
                    if target_norm in fr_name or fr_name in target_norm:
                        match_method = "Name Search"
                    else:
                        match = None # Discard low-confidence search result
                else:
                    match = None

        if match:
            fr_lei = match.get('lei', '')
            fr_id = match.get('id', '')
            fr_name = match.get('name', '')
            print(f"  => SUCCESS! Matched via {match_method} to FR ID {fr_id} (LEI: {fr_lei}, Name: {fr_name})")
            results.append({
                'target_lei': lei,
                'target_name': name,
                'fr_lei': fr_lei,
                'fr_id': fr_id,
                'fr_name': fr_name,
                'match_method': match_method
            })
        else:
            print("  => FAILED to find a match.")

    print(f"\nReconciliation complete. Successfully matched {len(results)} / {len(missing_companies)} companies.")
    
    if results:
        print(f"Saving results to {OUTPUT_MAP_PATH}")
        with open(OUTPUT_MAP_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['target_lei', 'target_name', 'fr_lei', 'fr_id', 'fr_name', 'match_method'])
            writer.writeheader()
            writer.writerows(results)

if __name__ == "__main__":
    main()
