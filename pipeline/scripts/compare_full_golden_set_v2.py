import csv
import sys
from pathlib import Path
import json

# Add pipeline to path
sys.path.insert(0, str(Path.cwd() / "pipeline"))
from src.markdown_chunker import chunk_markdown

metadata_path = Path("data/FinancialReports_downloaded/metadata.csv")
markdown_dir = Path("data/FinancialReports_downloaded/markdown")

golden_companies = [
    "AstraZeneca plc", "Aviva plc", "BAE Systems plc", "BT Group plc", 
    "Clarkson plc", "Croda International plc", "FirstGroup plc", 
    "Lloyds Banking Group plc", "National Grid plc", "Rolls-Royce Holdings plc", 
    "Schroders plc", "Serco Group plc", "Severn Trent plc", "Shell plc", "Tesco plc"
]

def find_filings(company_name):
    filings = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if company_name.lower() in row['company__name'].lower():
                filings.append(row)
    return filings

results = []
total_fr_chunks = 0

for company in golden_companies:
    filings = find_filings(company)
    # Filter for Annual Report (ESEF) preferred
    esef = [f for f in filings if "ESEF" in f['filing_type__name']]
    others = [f for f in filings if "ESEF" not in f['filing_type__name']]
    
    picked = {}
    for f in esef + others:
        title = f['title']
        ryear = None
        if "2023" in title: ryear = 2023
        elif "2024" in title: ryear = 2024
        elif "2022" in title: ryear = 2022
        
        if not ryear:
            ryear = int(f['release_datetime'][:4]) - 1
            
        if ryear not in picked and ryear in [2022, 2023]:
            picked[ryear] = f
            
    company_chunks = 0
    for year, f in picked.items():
        md_path = markdown_dir / f"{f['pk']}.md"
        if md_path.exists():
            with open(md_path, 'r', encoding='utf-8') as file:
                content = file.read()
            chunks = chunk_markdown(content, f['pk'], company, company, year)
            company_chunks += len(chunks)
            total_fr_chunks += len(chunks)
    
    results.append((company, company_chunks))

print(f"Total Chunks in Existing Golden Set: 474")
print(f"Total Chunks from FR Database: {total_fr_chunks}")
print("\nBreakdown by Company (Total for 2 years):")
for name, count in results:
    print(f"{name}: {count}")
