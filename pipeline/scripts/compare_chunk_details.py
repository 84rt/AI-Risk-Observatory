import sys
from pathlib import Path
import json

# Add pipeline to path
sys.path.insert(0, str(Path.cwd() / "pipeline"))

from src.markdown_chunker import chunk_markdown

def get_chunk_info(chunks):
    info = []
    for c in chunks:
        # Extract keywords and a snippet of the text
        info.append({
            "keywords": sorted(c['matched_keywords']),
            "text_snippet": c['chunk_text'][:100].replace('\n', ' ')
        })
    return info

def test_chunking(file_path, label):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    chunks = chunk_markdown(
        content,
        document_id=f"test-{label}",
        company_id="SHEL",
        company_name="Shell plc",
        report_year=2023
    )
    return chunks

fr_file = "data/FinancialReports_downloaded/markdown/2004721.md"
our_file = "data/processed/gs-phase1-20260117-152257/documents/None-2023-energy--extraction-ixbrl.md"

fr_chunks = test_chunking(fr_file, "FinancialReports")
our_chunks = test_chunking(our_file, "Our_iXBRL")

print(f"--- FinancialReports Chunks ({len(fr_chunks)}) ---")
for i, c in enumerate(get_chunk_info(fr_chunks)):
    print(f"{i+1}: {c['keywords']} | {c['text_snippet']}...")

print(f"\n--- Our_iXBRL Chunks ({len(our_chunks)}) ---")
for i, c in enumerate(get_chunk_info(our_chunks)):
    print(f"{i+1}: {c['keywords']} | {c['text_snippet']}...")
