import sys
from pathlib import Path
import json

# Add pipeline to path
sys.path.insert(0, str(Path.cwd() / "pipeline"))
from src.markdown_chunker import chunk_markdown

def dump_chunks(file_path, output_path, label):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    chunks = chunk_markdown(
        content,
        document_id=f"{label}-AVIVA-2023",
        company_id="AVIVA",
        company_name="Aviva plc",
        report_year=2023
    )
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"=== CHUNKS FOR {label} (Total: {len(chunks)}) ===\n\n")
        for i, c in enumerate(chunks):
            f.write(f"--- CHUNK {i+1} ---\n")
            f.write(f"Keywords: {c['matched_keywords']}\n")
            f.write(f"Sections: {c['report_sections']}\n")
            f.write(f"Text:\n{c['chunk_text']}\n")
            f.write("-" * 80 + "\n\n")
    
    return len(chunks)

# Source paths
our_path = "data/processed/gs-phase1-20260117-152257/documents/None-2023-insurance-ixbrl.md"
fr_path = "data/FinancialReports_downloaded/markdown/2001192.md"

# Output paths
output_dir = Path("data/results/comparison_audit")
output_dir.mkdir(parents=True, exist_ok=True)

our_out = output_dir / "aviva_2023_our_chunks.txt"
fr_out = output_dir / "aviva_2023_fr_chunks.txt"

n_our = dump_chunks(our_path, our_out, "OUR_iXBRL")
n_fr = dump_chunks(fr_path, fr_out, "FINANCIAL_REPORTS")

print(f"Our iXBRL Chunks: {n_our} -> {our_out}")
print(f"FR Database Chunks: {n_fr} -> {fr_out}")
2001192