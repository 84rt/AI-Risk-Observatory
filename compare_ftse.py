import csv

# Load constituents (the "source of truth" for 350)
constituents = {}
with open("ftse350_constituents.csv", "r") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for row in reader:
        name = row[0].lower()
        ticker = row[1].upper()
        constituents[ticker] = name

# Load our matched dataset
matched = {}
with open("data/FR_dataset/companies_with_lei.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ticker = row["ticker"].upper()
        matched[ticker] = row["name"].lower()

missing = []
for ticker, name in constituents.items():
    if ticker not in matched:
        missing.append((ticker, name))

print(f"Total Constituents: {len(constituents)}")
print(f"Matched by Ticker: {len(constituents) - len(missing)}")
print(f"Missing Tickers: {len(missing)}")
print("\nSample Missing:")
for t, n in missing[:10]:
    print(f"{t}: {n}")
