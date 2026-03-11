import csv
from pathlib import Path

lei_to_filings = {}
with open("data/FR_dataset/manifest.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lei = row["company__lei"]
        lei_to_filings[lei] = lei_to_filings.get(lei, 0) + 1

segments = {}
total_per_segment = {}
with_filings_per_segment = {}

with open("data/FR_dataset/companies_with_lei.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lei = row["lei"]
        seg = row["market_segment"]
        total_per_segment[seg] = total_per_segment.get(seg, 0) + 1
        if lei in lei_to_filings:
            with_filings_per_segment[seg] = with_filings_per_segment.get(seg, 0) + 1

print(f"{'Segment':<15} | {'Total':<10} | {'With Filings':<15} | {'Coverage':<10}")
print("-" * 55)
total_all = 0
total_with = 0
for seg in sorted(total_per_segment.keys()):
    t = total_per_segment[seg]
    w = with_filings_per_segment.get(seg, 0)
    total_all += t
    total_with += w
    print(f"{seg:<15} | {t:<10} | {w:<15} | {(w/t)*100:>8.1f}%")

print("-" * 55)
print(f"{'TOTAL':<15} | {total_all:<10} | {total_with:<15} | {(total_with/total_all)*100:>8.1f}%")
