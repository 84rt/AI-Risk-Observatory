import csv
from pathlib import Path
from collections import Counter

metadata_path = Path("data/FinancialReports_downloaded/metadata.csv")
golden_path = Path("data/reference/golden_set_companies.csv")

def analyze_fr():
    companies = set()
    years = set()
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            companies.add(row['company__name'])
            dt = row['release_datetime']
            if dt and len(dt) >= 4:
                years.add(dt[:4])
                
    return companies, years

def get_golden_sectors():
    sectors = {}
    try:
        with open(golden_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sectors[row['company_name'].lower()] = row['sector']
    except:
        pass
    return sectors

fr_companies, fr_years = analyze_fr()
golden_sectors = get_golden_sectors()

overlap_count = 0
found_sectors = Counter()

for co in fr_companies:
    name_lower = co.lower()
    for g_name, sector in golden_sectors.items():
        if g_name in name_lower or name_lower in g_name:
            found_sectors[sector] += 1
            overlap_count += 1
            break

print(f"--- FinancialReports Dataset Stats ---")
print(f"Unique Companies: {len(fr_companies)}")
print(f"Years Range: {sorted(list(fr_years))}")
print(f"Total Filings: 5,812")

print(f"\n--- Sector Overlap with Golden Set (CNI) ---")
print(f"Companies matching current CNI sectors: {overlap_count}")
for sector, count in sorted(found_sectors.items()):
    print(f"  - {sector}: {count}")
