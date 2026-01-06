#!/usr/bin/env python3
"""Quick test of the updated pipeline with a single company (Shell)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.xbrl_filings_client import XBRLFilingsClient
from src.ixbrl_extractor import iXBRLExtractor
from src.chunker import TextChunker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test with Shell plc
COMPANY = {
    "name": "Shell plc",
    "ticker": "SHEL",
    "lei": "21380068P1DRHMJ8KU70",
    "sector": "Oil & Gas"
}

def test_single_company():
    """Test the full workflow with one company."""

    print("=" * 80)
    print(f"Testing Pipeline with: {COMPANY['name']}")
    print("=" * 80)

    # Step 1: Download iXBRL report
    print("\n1. Downloading iXBRL report from filings.xbrl.org...")
    xbrl_client = XBRLFilingsClient()
    output_dir = settings.results_dir / "test_pipeline_single"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = xbrl_client.fetch_annual_report(
        lei=COMPANY["lei"],
        entity_name=COMPANY["name"],
        output_dir=output_dir
    )

    if not result:
        print("   ❌ Download failed")
        return

    report_path = Path(result["path"])
    print(f"   ✅ Downloaded: {report_path}")
    print(f"   Size: {report_path.stat().st_size / (1024*1024):.1f} MB")

    # Step 2: Extract text
    print("\n2. Extracting text from iXBRL report...")
    extractor = iXBRLExtractor()

    try:
        extracted = extractor.extract_report(report_path)
        print(f"   ✅ Extracted text:")
        print(f"   - Sections: {len(extracted.sections)}")
        print(f"   - Spans: {extracted.metadata.get('num_spans', 'N/A')}")
        print(f"   - Total text: {len(extracted.full_text):,} characters")

        # Show first few sections
        print(f"\n   First 5 sections:")
        for i, section_name in enumerate(list(extracted.sections.keys())[:5]):
            span_count = len(extracted.sections[section_name])
            print(f"      {i+1}. {section_name}: {span_count} spans")

    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return

    # Step 3: Chunk into candidates
    print("\n3. Chunking text into candidate spans...")
    chunker = TextChunker(chunk_by="paragraph")

    try:
        candidates = chunker.chunk_report(
            extracted_report=extracted,
            firm_id=COMPANY["ticker"],
            firm_name=COMPANY["name"],
            sector=COMPANY["sector"],
            report_year=2022
        )

        print(f"   ✅ Generated {len(candidates)} candidate spans")

        if candidates:
            sample = candidates[0]
            print(f"\n   Sample candidate:")
            print(f"      Firm: {sample.firm_id}")
            print(f"      Sector: {sample.sector}")
            print(f"      Text length: {len(sample.text)} chars")
            print(f"      Preview: {sample.text[:200]}...")

    except Exception as e:
        print(f"   ❌ Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("✅ Pipeline test successful!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run full pipeline: python pipeline_test.py")
    print("  2. Or test download only: python test_download_only.py")

if __name__ == "__main__":
    test_single_company()
