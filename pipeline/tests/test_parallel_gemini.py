#!/usr/bin/env python3
"""Test optimized parallel Gemini text cleaning."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.ixbrl_extractor import iXBRLExtractor
from src.preprocessor import Preprocessor, PreprocessingStrategy

print("=" * 80)
print("PARALLEL GEMINI TEXT CLEANING TEST")
print("=" * 80)

# Find a sample iXBRL file
ixbrl_dir = settings.raw_dir / "ixbrl"
ixbrl_files = list(ixbrl_dir.glob("*.xhtml"))

if not ixbrl_files:
    print("❌ No iXBRL files found")
    sys.exit(1)

# Sort by size to get smallest first
ixbrl_files_with_size = [(f, f.stat().st_size) for f in ixbrl_files]
ixbrl_files_with_size.sort(key=lambda x: x[1])

test_file, file_size = ixbrl_files_with_size[0]
company_name = test_file.stem.split("_")[0]

print(f"\n📄 Testing: {test_file.name}")
print(f"   Size: {file_size / (1024*1024):.1f} MB")
print("=" * 80)

# Extract
print("\n1️⃣ Extracting text from iXBRL...")
start = time.time()
extractor = iXBRLExtractor()
extracted = extractor.extract_report(test_file)
extract_time = time.time() - start
print(f"   ✅ Extracted {len(extracted.spans)} spans ({len(extracted.full_text):,} chars)")
print(f"   ⏱️  Extraction took: {extract_time:.1f}s")

# Preprocess WITHOUT cleaning
print("\n2️⃣ Preprocessing WITHOUT Gemini cleaning...")
start = time.time()
preprocessor_no_clean = Preprocessor(
    strategy=PreprocessingStrategy.RISK_ONLY,
    clean_text=False
)
preprocessed_no_clean = preprocessor_no_clean.process(extracted, company_name)
no_clean_time = time.time() - start
print(f"   ✅ Processed: {len(preprocessed_no_clean.markdown_content):,} chars")
print(f"   ⏱️  Processing took: {no_clean_time:.1f}s")
print(f"   📊 Retention: {preprocessed_no_clean.stats.get('retention_pct', 0):.1f}%")

# Preprocess WITH parallel Gemini cleaning
print("\n3️⃣ Preprocessing WITH parallel Gemini cleaning...")
print("   🚀 Using optimized settings:")
print("      - 100K character chunks (vs old 20K)")
print("      - 20 parallel workers")
print("      - 4 requests/second rate limit")
start = time.time()
preprocessor_clean = Preprocessor(
    strategy=PreprocessingStrategy.RISK_ONLY,
    clean_text=True
)
preprocessed_clean = preprocessor_clean.process(extracted, company_name)
clean_time = time.time() - start
print(f"   ✅ Processed: {len(preprocessed_clean.markdown_content):,} chars")
print(f"   ⏱️  Processing took: {clean_time:.1f}s")
print(f"   📊 Retention: {preprocessed_clean.stats.get('retention_pct', 0):.1f}%")

# Show comparison
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)
print(f"Extraction:              {extract_time:>6.1f}s")
print(f"Preprocessing (no clean): {no_clean_time:>6.1f}s")
print(f"Preprocessing (w/ clean): {clean_time:>6.1f}s")
print(f"Cleaning overhead:        {clean_time - no_clean_time:>6.1f}s")
print(f"Total time with cleaning: {extract_time + clean_time:>6.1f}s")
print("=" * 80)

# Show sample
print("\n4️⃣ Sample comparison:")
print("\nWithout cleaning (300 chars):")
print(preprocessed_no_clean.markdown_content[:300])
print("\nWith cleaning (300 chars):")
print(preprocessed_clean.markdown_content[:300])

print("\n✅ Test complete!")
print("\nNOTE: Cleaning is now ENABLED by default in pipeline_test.py")
print("      Use --no-clean-text flag to disable if needed for speed")
