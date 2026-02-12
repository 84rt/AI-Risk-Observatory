import sys
import os
from pathlib import Path
import json

# Add pipeline to path
sys.path.insert(0, str(Path.cwd() / "pipeline"))

from src.markdown_chunker import chunk_markdown

def get_file_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "chars": len(content),
        "kb": os.path.getsize(path) / 1024,
        "content": content
    }

# File paths
fr_path = "data/FinancialReports_downloaded/markdown/2004721.md"
our_path = "data/processed/gs-phase1-20260117-152257/documents/None-2023-energy--extraction-ixbrl.md"

# 1. Stats Comparison
fr_stats = get_file_stats(fr_path)
our_stats = get_file_stats(our_path)

# 2. Run Chunking
fr_chunks = chunk_markdown(
    fr_stats["content"],
    document_id="SHEL-2023-FR",
    company_id="SHEL",
    company_name="Shell plc",
    report_year=2023
)

our_chunks = chunk_markdown(
    our_stats["content"],
    document_id="SHEL-2023-OUR",
    company_id="SHEL",
    company_name="Shell plc",
    report_year=2023
)

# 3. Save Chunks
output_dir = Path("data/results/comparison_audit")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "fr_chunks_shell_2023.json", "w") as f:
    json.dump(fr_chunks, f, indent=2)
with open(output_dir / "our_chunks_shell_2023.json", "w") as f:
    json.dump(our_chunks, f, indent=2)

# 4. Report results
print(f"{'Metric':<25} | {'FinancialReports':<20} | {'Our iXBRL':<20}")
print("-" * 75)
print(f"{'File Size (KB)':<25} | {fr_stats['kb']:>18.2f} | {our_stats['kb']:>18.2f}")
print(f"{'Character Count':<25} | {fr_stats['chars']:>18,} | {our_stats['chars']:>18,}")
print(f"{'Total Chunks Found':<25} | {len(fr_chunks):>18} | {len(our_chunks):>18}")

# 5. Side-by-Side Example (Emerging Risks)
def find_emerging_risk_chunk(chunks):
    for c in chunks:
        if "emerging risk" in c['chunk_text'].lower() and "artificial intelligence" in c['chunk_text'].lower():
            return c
    return None

fr_er = find_emerging_risk_chunk(fr_chunks)
our_er = find_emerging_risk_chunk(our_chunks)

if fr_er and our_er:
    print("\n" + "="*80)
    print("SIDE-BY-SIDE: EMERGING RISKS CHUNK")
    print("="*80)
    print("\n[FINANCIAL REPORTS VERSION]")
    print("-" * 40)
    print(fr_er['chunk_text'])
    print("\n[OUR iXBRL VERSION]")
    print("-" * 40)
    print(our_er['chunk_text'])
    
    print(f"\nKeywords (FR): {fr_er['matched_keywords']}")
    print(f"Keywords (Our): {our_er['matched_keywords']}")
