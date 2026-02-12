
import sys
from pathlib import Path
import json

# Add pipeline to path
sys.path.insert(0, str(Path.cwd() / "pipeline"))
from src.markdown_chunker import chunk_markdown

# Test files from our own iXBRL processing
processed_dir = Path("data/processed/gs-phase1-20260117-152257/documents")
test_files = [
    {"name": "Aviva plc", "file": "None-2023-insurance-ixbrl.md", "year": 2023},
    {"name": "Aviva plc", "file": "None-2024-insurance-ixbrl.md", "year": 2024},
    {"name": "Severn Trent plc", "file": "None-2023-water-ixbrl.md", "year": 2023},
    {"name": "Severn Trent plc", "file": "None-2024-water-ixbrl.md", "year": 2024},
]

print(f"{'Company':<20} | {'Year':<5} | {'Chunks Found (Current Alg)'}")
print("-" * 50)

for item in test_files:
    path = processed_dir / item['file']
    if not path.exists():
        print(f"File not found: {path}")
        continue
        
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    chunks = chunk_markdown(
        content, 
        document_id="diag", 
        company_id="DIAG", 
        company_name=item['name'], 
        report_year=item['year']
    )
    print(f"{item['name']:<20} | {item['year']:<5} | {len(chunks)}")
