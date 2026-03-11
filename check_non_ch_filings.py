import csv

def normalize(n):
    return n.lower().replace(" plc", "").replace(" ltd", "").replace(" group", "").replace(".", "").replace(",", "").strip()

# 1. Load CH Coverage (to identify who is NOT in CH)
ch_names = set()
with open("data/FR_dataset/ch_coverage.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["ch_found"] == "True":
            ch_names.add(normalize(row["name"]))

# 2. Load FR Manifest (to see who has filings)
fr_names_with_filings = set()
with open("data/FR_dataset/manifest.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fr_names_with_filings.add(normalize(row["company__name"]))

# 3. Load FTSE 350 Constituents
constituents = []
with open("ftse350_constituents.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        constituents.append({"name": row[0], "ticker": row[1]})

not_in_ch = []
for c in constituents:
    if normalize(c["name"]) not in ch_names:
        not_in_ch.append(c)

print(f"Total FTSE 350: {len(constituents)}")
print(f"Not in Companies House: {len(not_in_ch)}")

has_fr_filings = []
no_fr_filings = []

for c in not_in_ch:
    if normalize(c["name"]) in fr_names_with_filings:
        has_fr_filings.append(c)
    else:
        no_fr_filings.append(c)

print(f"Of the {len(not_in_ch)} not in CH:")
print(f"  - Have filings in our FR dataset: {len(has_fr_filings)}")
print(f"  - Missing filings in our FR dataset: {len(no_fr_filings)}")

print("\n--- Non-CH Companies WITH FR Filings ---")
for c in has_fr_filings[:10]:
    print(f"{c['ticker']}: {c['name']}")

print("\n--- Non-CH Companies MISSING FR Filings (Sample) ---")
for c in no_fr_filings[:10]:
    print(f"{c['ticker']}: {c['name']}")
