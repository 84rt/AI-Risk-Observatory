import csv
from pathlib import Path

# Paths
metadata_path = Path("data/FinancialReports_downloaded/metadata.csv")
output_path = Path("pipeline/data/selected_candidates_temp.csv")

# Your specific list of 35 companies
target_companies = [
    "ASSOCIATED BRITISH ENGINEERING PLC",
    "BSF ENTERPRISE PLC",
    "Babcock International Group PLC",
    "Barclays PLC",
    "Barratt Developments PLC",
    "British Land Co PLC",
    "Capita PLC",
    "Carnival PLC",
    "Centrica PLC",
    "Crest Nicholson Holdings PLC",
    "GlaxoSmithKline PLC",
    "HSBC Holdings PLC",
    "IntegraFin Holdings PLC",
    "Kingfisher PLC",
    "LOWLAND INV CO PLC",
    "London Stock Exchange Group PLC",
    "Natwest Group PLC",
    "Paragon Banking Group PLC",
    "Prudential PLC",
    "R.E.A. HOLDINGS PLC",
    "RICARDO PLC",
    "Relx PLC",
    "SSE PLC",
    "SSP Group PLC",
    "Safestore Holdings PLC",
    "Sage Group PLC",
    "Sainsbury (J) PLC",
    "Segro PLC",
    "Standard Chartered PLC",
    "Tritax Eurobox PLC",
    "Unilever PLC",
    "United Utilities Group PLC",
    "Victrex PLC",
    "Vodafone Group PLC",
    "Whitbread PLC"
]

def map_targets():
    # 1. Load metadata into a lookup dict (Name -> LEI/Data)
    fr_lookup = {}
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize name for matching
            raw_name = row.get('company__name', '').strip()
            norm_name = raw_name.lower().replace("  ", " ")
            
            # Store first valid LEI we find for this name
            if norm_name not in fr_lookup:
                fr_lookup[norm_name] = {
                    "lei": row.get('company__lei', ''),
                    "years": set(),
                    "raw_name": raw_name
                }
            
            # Track years available
            dt = row.get('release_datetime', '')
            if dt and len(dt) >= 4:
                try:
                    year = int(dt[:4])
                    if year in [2022, 2023, 2024, 2025]:
                        fr_lookup[norm_name]["years"].add(year)
                except:
                    pass

    # 2. Match targets to FR data
    mapped_candidates = []
    missing = []

    print(f"Mapping {len(target_companies)} companies...")
    print("-" * 60)

    for target in target_companies:
        target_norm = target.lower().replace("  ", " ")
        match = None
        
        # Exact match first
        if target_norm in fr_lookup:
            match = fr_lookup[target_norm]
        else:
            # Fuzzy / Substring match
            for fr_name, data in fr_lookup.items():
                if target_norm in fr_name or fr_name in target_norm:
                    # Basic safeguard: ensure "PLC" isn't the only match
                    if len(target_norm) > 10: 
                        match = data
                        break
        
        if match:
            print(f"Found: {target:<35} -> {match['raw_name']} (LEI: {match['lei']}) Years: {sorted(list(match['years']))}")
            mapped_candidates.append({
                "name": match['raw_name'],
                "lei": match['lei'],
                "years": sorted(list(match['years'])),
                "score": 1 # Placeholder
            })
        else:
            print(f"MISSING: {target}")
            missing.append(target)

    # 3. Save to temp CSV (compatible with build_full_manifest.py)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "lei", "years", "score"])
        writer.writeheader()
        for c in mapped_candidates:
            writer.writerow(c)
            
    print(f"\nSaved {len(mapped_candidates)} mapped companies to {output_path}")
    if missing:
        print(f"Warning: {len(missing)} companies could not be found in local FR database.")

if __name__ == "__main__":
    map_targets()
