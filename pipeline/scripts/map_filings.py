import csv
from pathlib import Path

metadata_path = Path("data/FinancialReports_downloaded/metadata.csv")

targets = [
    "HSBC Holdings PLC", "Standard Chartered PLC", "Abrdn PLC", 
    "British American Tobacco PLC", "BP PLC", "Unilever PLC", 
    "Prudential PLC", "3i Group PLC", "Segro PLC", "Natwest Group PLC", 
    "WPP PLC", "Safestore Holdings PLC", "Relx PLC", "Paragon Banking Group PLC", 
    "Intercontinental Hotels Group PLC", "Haleon PLC", "Carnival PLC", 
    "British Land Co PLC", "Barratt Developments PLC", "Tritax Big Box REIT PLC", 
    "Rio Tinto PLC", "Next PLC", "Imperial Brands PLC", "GlaxoSmithKline PLC", 
    "Games Workshop Group PLC", "Beazley PLC", "Volution Group PLC", 
    "M&G PLC", "Land Securities Group PLC", "JD Sports Fashion PLC", 
    "Diageo PLC", "Currys PLC", "Computacenter PLC", "Coats Group PLC", 
    "Anglo American PLC"
]

def find_filings(company_name):
    filings = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if company_name.lower() in row['company__name'].lower():
                date_str = row['release_datetime']
                if not date_str: continue
                year = date_str[:4]
                if year in ["2023", "2024", "2025"]:
                    filings.append(row)
    return filings

results = {}
for target in targets:
    filings = find_filings(target)
    esef = [f for f in filings if "ESEF" in f['filing_type__name']]
    others = [f for f in filings if "ESEF" not in f['filing_type__name']]
    
    picked = {}
    for f in esef + others:
        title = f['title']
        ryear = None
        if "2023" in title: ryear = "2023"
        elif "2022" in title: ryear = "2022"
        elif "2024" in title: ryear = "2024"
        
        if not ryear:
            ryear = str(int(f['release_datetime'][:4]) - 1)
            
        if ryear not in picked and ryear in ["2022", "2023", "2024"]:
            picked[ryear] = f
            
    results[target] = picked

found_count = 0
for target, picked in results.items():
    years = sorted(picked.keys())
    print(f"{target}: {years}")
    if len(years) >= 2:
        found_count += 1

print(f"\nTotal companies with at least 2 years of reports: {found_count}")
