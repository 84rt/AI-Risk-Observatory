#!/usr/bin/env python3
"""Diagnose iXBRL section detection for a sample file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ixbrl_extractor import iXBRLExtractor

# Find a sample iXBRL file
ixbrl_dir = Path("output/reports/ixbrl")
ixbrl_files = list(ixbrl_dir.glob("*.xhtml"))

if not ixbrl_files:
    print("❌ No iXBRL files found")
    sys.exit(1)

# Use Tesco as example
tesco_file = None
for f in ixbrl_files:
    if "Tesco" in f.name:
        tesco_file = f
        break

if not tesco_file:
    tesco_file = ixbrl_files[0]

print(f"📄 Analyzing: {tesco_file.name}")
print("=" * 80)

# Extract
extractor = iXBRLExtractor()
extracted = extractor.extract_report(tesco_file)

print(f"\n✅ Extracted {len(extracted.spans)} spans")

# Check sections
sections = extracted.sections
print(f"✅ Found {len(sections)} sections")
print()

print("📋 SECTIONS:")
print("=" * 80)
for section_name, spans in sections.items():
    print(f"  {section_name}: {len(spans)} spans")

print()
print("🔍 Looking for headings with 'risk' keyword...")
print("=" * 80)

risk_headings = [
    span for span in extracted.spans[:500]  # Check first 500 spans
    if span.is_heading and 'risk' in span.text.lower()
]

if risk_headings:
    print(f"\n✅ Found {len(risk_headings)} headings containing 'risk':")
    for h in risk_headings[:10]:
        print(f"  Section: {h.section or 'None'}")
        print(f"  Text: {h.text[:80]}")
        print()
else:
    print("⚠️  No headings with 'risk' found in first 500 spans")

# Show sample headings
print("\n🔍 Sample of ALL headings (first 20):")
print("=" * 80)
headings = [s for s in extracted.spans[:1000] if s.is_heading]
for h in headings[:20]:
    print(f"  Section: {h.section or 'None':30s} | {h.text[:60]}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
