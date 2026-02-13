import csv
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
base_dir = Path(__file__).parent.parent.parent
load_dotenv(base_dir / ".env.local")

FR_API_KEY = os.getenv("FR_API_KEY")
GOLDEN_CSV = base_dir / "data" / "reference" / "golden_set_companies_with_lei.csv"
CANDIDATES_CSV = base_dir / "pipeline" / "data" / "selected_candidates_temp.csv"
OUTPUT_CSV = base_dir / "data" / "reference" / "companies_metadata_v2.csv"

# ISIC Cache
isic_name_cache = {}

def get_isic_name(code):
    if not code:
        return "Unknown"
    if code in isic_name_cache:
        return isic_name_cache[code]
    
    url = "https://api.financialreports.eu/isic-classes/"
    headers = {"X-API-Key": FR_API_KEY}
    params = {"code": code}
    
    try:
        print(f"  Fetching ISIC name for code {code}...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            name = data["results"][0].get("name", "Unknown")
            isic_name_cache[code] = name
            time.sleep(0.2)
            return name
    except:
        pass
    
    return f"Code: {code}"

def get_company_details(lei):
    if not lei or lei == "None" or lei == "Unknown":
        return None
    
    url = "https://api.financialreports.eu/companies/"
    headers = {"X-API-Key": FR_API_KEY}
    params = {"lei": lei}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            return data["results"][0]
    except Exception as e:
        print(f"Error fetching data for {lei}: {e}")
    
    return None

def main():
    if not FR_API_KEY:
        print("FR_API_KEY not found in .env.local")
        return

    all_companies = []
    processed_leis = set()

    # 1. Process Golden Set (15)
    print("Processing Golden Set (15)...")
    if GOLDEN_CSV.exists():
        with open(GOLDEN_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lei = row.get("lei")
                if not lei or lei in processed_leis:
                    continue
                
                fr_data = get_company_details(lei)
                isic_code = ""
                isic_name = ""
                
                if fr_data:
                    isic_code = fr_data.get("sub_industry_code", "")
                    if isic_code:
                        isic_name = get_isic_name(isic_code)
                
                all_companies.append({
                    "company_name": row["company_name"],
                    "lei": lei,
                    "cni_sector": row["sector"],
                    "isic_sector_code": isic_code,
                    "isic_sector_name": isic_name,
                    "source_type": "Golden Set"
                })
                processed_leis.add(lei)
                time.sleep(0.2)

    # 2. Process New Candidates (35)
    print("\nProcessing New Candidates (35)...")
    if CANDIDATES_CSV.exists():
        with open(CANDIDATES_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lei = row["lei"]
                if not lei or lei in processed_leis:
                    continue
                
                fr_data = get_company_details(lei)
                isic_code = ""
                isic_name = ""
                
                if fr_data:
                    isic_code = fr_data.get("sub_industry_code", "")
                    if isic_code:
                        isic_name = get_isic_name(isic_code)
                
                all_companies.append({
                    "company_name": row["name"],
                    "lei": lei,
                    "cni_sector": "Other",
                    "isic_sector_code": isic_code,
                    "isic_sector_name": isic_name,
                    "source_type": "Expansion"
                })
                processed_leis.add(lei)
                time.sleep(0.2)

    # 3. Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["company_name", "lei", "cni_sector", "isic_sector_code", "isic_sector_name", "source_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in all_companies:
            writer.writerow(c)

    print(f"\nSaved {len(all_companies)} companies to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
