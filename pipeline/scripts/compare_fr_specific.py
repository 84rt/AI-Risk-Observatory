
import csv
import sys
from pathlib import Path
import json

# Add pipeline to path
sys.path.insert(0, str(Path.cwd() / "pipeline"))
from src.markdown_chunker import chunk_markdown

metadata_path = Path("data/FinancialReports_downloaded/metadata.csv")
markdown_dir = Path("data/FinancialReports_downloaded/markdown")

test_targets = ["Aviva PLC", "Severn Trent PLC"]

def find_filings(company_name):
    filings = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if company_name.lower() in row['company__name'].lower():
                filings.append(row)
    return filings

print(f"{'Company':<20} | {'Year':<5} | {'FR Chunks (Current Alg)'}")
print("-" * 50)

for target in test_targets:
    filings = find_filings(target)
    # Prefer ESEF
    esef = [f for f in filings if "ESEF" in f['filing_type__name']]
    others = [f for f in filings if "ESEF" not in f['filing_type__name']]
    
    picked = {}
    for f in esef + others:
        ryear = None
        if "2023" in f['title']: ryear = 2023
        elif "2024" in f['title']: ryear = 2024
        elif "2022" in f['title']: ryear = 2022
        if not ryear: ryear = int(f['release_datetime'][:4]) - 1
        
        if ryear not in picked and ryear in [2022, 2023]:
            picked[ryear] = f
            
    for year, f in picked.items():
        md_path = markdown_dir / f"{f['pk']}.md"
        if md_path.exists():
            with open(md_path, 'r', encoding='utf-8') as file:
                content = file.read()
            chunks = chunk_markdown(content, f['pk'], target, target, year)
            print(f"{target:<20} | {year:<5} | {len(chunks)}")
