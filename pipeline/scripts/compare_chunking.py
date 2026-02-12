import sys
from pathlib import Path
import json

# Add pipeline to path
sys.path.insert(0, str(Path.cwd() / "pipeline"))

from src.markdown_chunker import chunk_markdown

def test_chunking(file_path, label):
    print(f"--- Testing Chunking for {label} ---")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    chunks = chunk_markdown(
        content,
        document_id=f"test-{label}",
        company_id="SHEL",
        company_name="Shell plc",
        report_year=2023
    )
    
    print(f"Total chunks found: {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i+1} (Sections: {chunk['report_sections']}):")
        text_preview = chunk['chunk_text'][:200].replace('\n', ' ')
        print(f"Text: {text_preview}...")
        print(f"Keywords: {chunk['matched_keywords']}")
    
    return chunks

fr_file = "data/FinancialReports_downloaded/markdown/2004721.md"
our_file = "data/processed/gs-phase1-20260117-152257/documents/None-2023-energy--extraction-ixbrl.md"

fr_chunks = test_chunking(fr_file, "FinancialReports")
print()
our_chunks = test_chunking(our_file, "Our_iXBRL")

# Compare keyword overlap
fr_keywords = set()
for c in fr_chunks:
    fr_keywords.update(c['matched_keywords'])

our_keywords = set()
for c in our_chunks:
    our_keywords.update(c['matched_keywords'])

print(f"\n--- Keyword Comparison ---")
print(f"FR Keywords: {sorted(fr_keywords)}")
print(f"Our Keywords: {sorted(our_keywords)}")
print(f"Intersection: {sorted(fr_keywords.intersection(our_keywords))}")
print(f"Only in FR: {sorted(fr_keywords - our_keywords)}")
print(f"Only in Our: {sorted(our_keywords - fr_keywords)}")
