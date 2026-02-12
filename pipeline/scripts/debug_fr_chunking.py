import sys
from pathlib import Path
import json
import re

# Add pipeline to path
sys.path.insert(0, str(Path.cwd() / "pipeline"))

from src.markdown_chunker import chunk_markdown, _parse_markdown_blocks, _extract_paragraphs

def debug_fr_chunking(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    blocks = _parse_markdown_blocks(content)
    paragraphs = _extract_paragraphs(blocks)
    
    # Find all mentions in paragraphs
    ai_mentions = []
    for p in paragraphs:
        if re.search(r"artificial intelligence", p.text, re.I):
            ai_mentions.append(p)
    
    print(f"Total paragraphs with 'artificial intelligence': {len(ai_mentions)}")
    for p in ai_mentions:
        print(f"Paragraph Index {p.index} (Block {p.block_index}, Section {p.section}):")
        print(f"Text Snippet: {p.text[:200]}...")
        
    chunks = chunk_markdown(
        content,
        document_id="debug-fr",
        company_id="SHEL",
        company_name="Shell plc",
        report_year=2023
    )
    
    print(f"\nTotal Chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1}: Paragraphs {c.get('paragraph_start')} to {c.get('paragraph_end')}")

debug_fr_chunking("data/FinancialReports_downloaded/markdown/2004721.md")
