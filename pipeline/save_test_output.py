#!/usr/bin/env python3
"""Save preprocessed output to files for inspection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ixbrl_extractor import iXBRLExtractor
from src.preprocessor import Preprocessor, PreprocessingStrategy

print("Creating sample preprocessed output for inspection...")

# Find RELX file (smallest)
ixbrl_dir = Path("output/reports/ixbrl")
ixbrl_files = list(ixbrl_dir.glob("*RELX*.xhtml"))

if not ixbrl_files:
    ixbrl_files = list(ixbrl_dir.glob("*549300WSX3VBUFFJOO66*.xhtml"))

if not ixbrl_files:
    print("❌ RELX file not found")
    sys.exit(1)

test_file = ixbrl_files[0]
company_name = "RELX PLC"

print(f"📄 Processing: {test_file.name}")

# Extract
extractor = iXBRLExtractor()
extracted = extractor.extract_report(test_file)
print(f"✅ Extracted {len(extracted.spans)} spans")

# Process with both strategies
for strategy in [PreprocessingStrategy.RISK_ONLY, PreprocessingStrategy.KEYWORD]:
    print(f"\n📊 Strategy: {strategy.value}")

    preprocessor = Preprocessor(
        strategy=strategy,
        include_context=True,
        clean_text=False
    )
    preprocessed = preprocessor.process(extracted, company_name)

    # Save to sample directory
    output_dir = Path("output/samples") / strategy.value
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"RELX_{strategy.value}.md"

    preprocessor.save_to_file(preprocessed, output_path)

    print(f"   ✅ Saved to: {output_path}")
    print(f"   📊 Size: {len(preprocessed.markdown_content):,} chars")
    print(f"   📊 Retention: {preprocessed.stats.get('retention_pct', 0):.1f}%")

print("\n✅ Sample files created in output/samples/")
print("\nView them with:")
print("  cat output/samples/risk_only/RELX_risk_only.md | head -100")
print("  cat output/samples/keyword/RELX_keyword.md | head -100")
