import csv
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
base_dir = Path(__file__).parent.parent.parent
load_dotenv(base_dir / ".env.local")

FR_API_KEY = os.getenv("FR_API_KEY")
GOLDEN_CSV = base_dir / "data" / "reference" / "golden_set_companies_with_lei.csv"

# Cache for ISIC names
isic_cache = {}

def get_isic_name(code):
    if not code:
        return "Unknown"
    if code in isic_cache:
        return isic_cache[code]
    
    url = f"https://api.financialreports.eu/isic-classes/"
    headers = {"X-API-Key": FR_API_KEY}
    params = {"code": code}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            name = data["results"][0].get("name", "Unknown")
            isic_cache[code] = name
            return name
    except:
        pass
    
    return f"Code: {code}"

def get_fr_company_data(lei):
    if not lei or lei == "None":
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

    print(f"{'Company Name':<30} | {'Our Sector':<25} | {'FR ISIC Sector (Class)':<40}")
    print("-" * 100)

    with open(GOLDEN_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["company_name"]
            our_sector = row["sector"]
            lei = row["lei"]
            
            fr_data = get_fr_company_data(lei)
            
            if fr_data:
                sub_code = fr_data.get("sub_industry_code")
                isic_name = get_isic_name(sub_code)
                print(f"{name[:30]:<30} | {our_sector[:25]:<25} | {isic_name[:40]:<40}")
            else:
                print(f"{name[:30]:<30} | {our_sector[:25]:<25} | {'Not Found in FR':<40}")

if __name__ == "__main__":
    main()
