import json
import csv
from pathlib import Path

# Load all FR companies
with open("all_fr_companies_debug.json", "r") as f:
    fr_data = json.load(f)

fr_tickers = {c.get("ticker").upper() for c in fr_data if c.get("ticker")}
fr_names = {c.get("name").lower() for c in fr_data if c.get("name")}

def normalize_name(n):
    return n.lower().replace(" plc", "").replace(" ltd", "").strip()

fr_names_norm = {normalize_name(n) for n in fr_names}

# Load Master list
master_path = Path("data/reference/ftse350_history/ftse350_company_master.csv")
master_total = 0
matches = 0

with open(master_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        master_total += 1
        ticker = row["canonical_ticker"].upper()
        name = row["canonical_company_name"].lower()
        norm_name = normalize_name(name)
        
        if ticker in fr_tickers or name in fr_names or norm_name in fr_names_norm:
            matches += 1

print(f"Master List Total: {master_total}")
print(f"Matches in FR: {matches}")
print(f"Overlap: {(matches/master_total)*100:.1f}%")
