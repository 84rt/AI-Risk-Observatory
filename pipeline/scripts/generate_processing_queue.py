import csv
import json
from pathlib import Path

# Paths
manifest_path = Path("data/reference/companies_metadata_v2.csv")
fr_metadata_path = Path("data/FinancialReports_downloaded/metadata.csv")
output_path = Path("data/processing_queue.json")

def generate_queue():
    # 1. Load Master Manifest
    companies = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lei = row['lei']
            companies[lei] = {
                "name": row['company_name'],
                "cni_sector": row['cni_sector'],
                "isic_code": row['isic_sector_code'],
                "isic_name": row['isic_sector_name'],
                "source_type": row['source_type']
            }
            
    print(f"Loaded {len(companies)} companies from manifest.")

    # 2. Scan FR Metadata for Filings
    filings_by_lei = {} # lei -> {year: filing_row}
    
    with open(fr_metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lei = row['company__lei']
            if lei not in companies:
                continue
                
            if lei not in filings_by_lei:
                filings_by_lei[lei] = {}
            
            # Determine year
            title = row['title']
            ryear = None
            if "2023" in title: ryear = 2023
            elif "2024" in title: ryear = 2024
            elif "2022" in title: ryear = 2022
            
            if not ryear and row['release_datetime']:
                try:
                    ryear = int(row['release_datetime'][:4]) - 1
                except:
                    pass
            
            if ryear in [2023, 2024]:
                # Priority logic: Prefer "ESEF" > "Annual Report" > Other
                current = filings_by_lei[lei].get(ryear)
                new_priority = 0
                if "ESEF" in row['filing_type__name']: new_priority = 3
                elif "Annual Report" in row['filing_type__name']: new_priority = 2
                else: new_priority = 1
                
                old_priority = 0
                if current:
                    if "ESEF" in current['filing_type__name']: old_priority = 3
                    elif "Annual Report" in current['filing_type__name']: old_priority = 2
                    else: old_priority = 1
                
                if new_priority > old_priority:
                    filings_by_lei[lei][ryear] = row
                elif new_priority == old_priority:
                    # Tie-break: prefer later release date (latest version)
                    if not current or row['release_datetime'] > current['release_datetime']:
                        filings_by_lei[lei][ryear] = row

    # 3. Build Queue
    queue = []
    missing_log = []
    
    for lei, info in companies.items():
        filings = filings_by_lei.get(lei, {})
        
        for year in [2023, 2024]:
            if year in filings:
                f = filings[year]
                md_path = Path("data/FinancialReports_downloaded/markdown") / f"{f['pk']}.md"
                
                if md_path.exists():
                    queue.append({
                        "company_name": info['name'],
                        "lei": lei,
                        "year": year,
                        "cni_sector": info['cni_sector'],
                        "isic_sector": info['isic_name'],
                        "file_path": str(md_path),
                        "pk": f['pk'],
                        "source_title": f['title']
                    })
                else:
                    missing_log.append(f"{info['name']} ({year}): Markdown file missing for PK {f['pk']}")
            else:
                missing_log.append(f"{info['name']} ({year}): No filing found in metadata")

    # 4. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)
        
    print(f"\nGenerated queue with {len(queue)} items.")
    print(f"Saved to {output_path}")
    
    if missing_log:
        print(f"\nMissing Reports ({len(missing_log)}):")
        for m in missing_log[:10]:
            print(f"  - {m}")
        if len(missing_log) > 10:
            print(f"  ... and {len(missing_log)-10} more.")

if __name__ == "__main__":
    generate_queue()
