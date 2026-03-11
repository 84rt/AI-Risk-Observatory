import csv

def normalize(n):
    return n.lower().replace(" plc", "").replace(" ltd", "").replace(" group", "").replace(".", "").replace(",", "").strip()

# Load our matched dataset
id_to_data = {}
name_to_id = {}
ticker_to_id = {}

with open("data/FR_dataset/companies_with_lei.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fid = row["id"]
        name = row["name"]
        ticker = row["ticker"]
        id_to_data[fid] = row
        name_to_id[normalize(name)] = fid
        ticker_to_id[ticker.upper()] = fid

# Load constituents
constituents = []
with open("ftse350_constituents.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        constituents.append({"name": row[0], "ticker": row[1]})

found = []
missing = []

for c in constituents:
    fid = ticker_to_id.get(c["ticker"].upper())
    if not fid:
        fid = name_to_id.get(normalize(c["name"]))
    
    if fid:
        found.append((c["ticker"], c["name"], fid))
    else:
        missing.append((c["ticker"], c["name"]))

print(f"Total Constituents: {len(constituents)}")
print(f"Found in Dataset: {len(found)}")
print(f"Missing from Dataset: {len(missing)}")

print(f"\n--- Breakdown of the {len(found)} Found ---")
segment_counts = {}
for t, n, fid in found:
    seg = id_to_data[fid]["market_segment"]
    segment_counts[seg] = segment_counts.get(seg, 0) + 1
for s, count in segment_counts.items():
    print(f"{s}: {count}")

print("\n--- Top 10 Missing ---")
for t, n in missing[:10]:
    print(f"{t}: {n}")
