import csv
import sys
import os
from pathlib import Path
import json

# Add pipeline to path
sys.path.insert(0, str(Path.cwd() / "pipeline"))
from src.markdown_chunker import chunk_markdown

metadata_path = Path("data/FinancialReports_downloaded/metadata.csv")
markdown_dir = Path("data/FinancialReports_downloaded/markdown")

golden_companies = [
    "Croda International", "Rolls-Royce Holdings", "BT Group", "BAE Systems",
    "Serco Group", "Shell", "Lloyds Banking Group", "Tesco", "AstraZeneca",
    "National Grid", "Severn Trent", "Aviva", "Schroders", "FirstGroup", "Clarkson"
]

def get_fr_filings():
    mappings = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['company__name'].lower()
            for target in golden_companies:
                if target.lower() in name:
                    date_val = row['release_datetime']
                    if not date_val: continue
                    year = date_val[:4]
                    if year in ["2023", "2024", "2025"]:
                        if target not in mappings: mappings[target] = []
                        mappings[target].append(row)
    return mappings

all_filings = get_fr_filings()

selected_pks = []
for company, filings in all_filings.items():
    picked = {}
    filings.sort(key=lambda x: "ESEF" in x['filing_type__name'], reverse=True)
    
    for f in filings:
        title = f['title']
        ryear = None
        if "2023" in title: ryear = 2023
        elif "2024" in title: ryear = 2024
        elif "2022" in title: ryear = 2022
        
        if not ryear:
            ryear = int(f['release_datetime'][:4]) - 1
            
        if ryear not in picked and ryear in [2022, 2023]:
            picked[ryear] = f
    
    for y, f in picked.items():
        selected_pks.append({
            "company": company,
            "year": y,
            "pk": f['pk']
        })

total_fr_chunks = 0
for item in selected_pks:
    md_path = markdown_dir / f"{item['pk']}.md"
    if not md_path.exists(): continue
        
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    chunks = chunk_markdown(content, item['pk'], item['company'], item['company'], item['year'])
    total_fr_chunks += len(chunks)

print(f"Total Chunks in Existing Golden Set: 474")
print(f"Total Chunks from FR Database: {total_fr_chunks}")
