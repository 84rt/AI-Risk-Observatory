import csv
from pathlib import Path
from collections import defaultdict

# Use correct relative paths
metadata_path = Path("data/FinancialReports_downloaded/metadata.csv")
golden_path = Path("data/reference/golden_set_companies.csv")

def get_golden_names():
    names = set()
    try:
        if golden_path.exists():
            with open(golden_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    names.add(row['company_name'].lower())
    except:
        pass
    return names

def select_candidates():
    golden_names = get_golden_names()
    
    # Group filings by company
    company_filings = defaultdict(set)
    company_leis = {}
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('company__name', '').strip()
            lei = row.get('company__lei', '').strip()
            dt = row.get('release_datetime', '')
            
            # Extract year
            if dt and len(dt) >= 4:
                try:
                    year = int(dt[:4])
                    # We want 2023 and 2024
                    if year in [2023, 2024]:
                        company_filings[name].add(year)
                        if lei:
                            company_leis[name] = lei
                except ValueError:
                    continue

    candidates = []
    
    for name, years in company_filings.items():
        # Skip if likely already in golden set (basic fuzzy match)
        is_golden = False
        name_lower = name.lower()
        for g in golden_names:
            if g in name_lower or name_lower in g:
                is_golden = True
                break
        
        if is_golden:
            continue
            
        has_2023 = 2023 in years
        has_2024 = 2024 in years
        
        score = 0
        if has_2023 and has_2024:
            score = 3
        elif has_2024:
            score = 2
        elif has_2023:
            score = 1
            
        if score > 0:
            candidates.append({
                "name": name,
                "lei": company_leis.get(name, "Unknown"),
                "score": score,
                "years": sorted(list(years))
            })
    
    # Sort by score (descending)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Select top 35
    top_35 = candidates[:35]
    
    print(f"Selected {len(top_35)} candidates.")
    print("-" * 60)
    for i, c in enumerate(top_35):
        print(f"{i+1:02d}. {c['name'][:35]:<35} | LEI: {c['lei']} | Years: {c['years']}")
    
    # Save to a temp CSV for the next step
    output_path = Path("pipeline/data/selected_candidates_temp.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "lei", "years", "score"])
        writer.writeheader()
        for c in top_35:
            writer.writerow(c)
            
    print(f"\nSaved candidates to {output_path}")

if __name__ == "__main__":
    select_candidates()
