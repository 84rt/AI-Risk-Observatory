#!/usr/bin/env python3
"""Diagnostic script to check section detection across all golden dataset companies.

This script:
1. Loads all downloaded reports
2. Extracts text and detects sections
3. Shows section names and sample content
4. Saves detailed reports for manual review
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from src.xbrl_filings_client import XBRLFilingsClient
from src.companies_house import CompaniesHouseClient
from src.ixbrl_extractor import iXBRLExtractor
from src.pdf_extractor import PDFExtractor
from src.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_companies():
    """Load companies with LEI codes."""
    settings = get_settings()
    lei_file = settings.data_dir / "reference" / "companies_with_lei.json"
    with open(lei_file, 'r') as f:
        return json.load(f)


def diagnose_all_reports():
    """Diagnose section detection for all downloaded reports."""
    companies = load_companies()
    settings = get_settings()
    reports_dir = settings.raw_dir
    ixbrl_dir = reports_dir / "ixbrl"
    pdf_dir = reports_dir / "pdfs"

    pdf_extractor = PDFExtractor()
    ixbrl_extractor = iXBRLExtractor()

    results = []

    print("\n" + "=" * 80)
    print("SECTION DETECTION DIAGNOSTIC REPORT")
    print("=" * 80)

    for company in companies:
        company_number = company["number"]
        company_name = company["name"]
        ticker = company["ticker"]

        # Find report
        report_path = None
        report_format = None

        if ixbrl_dir.exists():
            matching_ixbrl = list(ixbrl_dir.glob(f"{company_number}_*.xhtml")) or \
                            list(ixbrl_dir.glob(f"*{ticker}*.xhtml"))
            if matching_ixbrl:
                report_path = matching_ixbrl[0]
                report_format = "ixbrl"

        if not report_path and pdf_dir.exists():
            matching_pdfs = list(pdf_dir.glob(f"{company_number}_*.pdf"))
            if matching_pdfs:
                report_path = matching_pdfs[0]
                report_format = "pdf"

        if not report_path:
            print(f"\n❌ {company_name}: No report found")
            results.append({
                "company": company_name,
                "status": "no_report"
            })
            continue

        print(f"\n{'='*80}")
        print(f"Company: {company_name} ({ticker})")
        print(f"Format: {report_format.upper()}")
        print(f"File: {report_path.name}")
        print(f"{'='*80}")

        try:
            # Extract
            if report_format == "ixbrl":
                extracted = ixbrl_extractor.extract_report(report_path)
            else:
                extracted = pdf_extractor.extract_report(report_path)

            # Get sections
            sections = extracted.sections

            print(f"\n📊 Extraction Summary:")
            print(f"   Total spans: {len(extracted.spans):,}")
            print(f"   Text length: {len(extracted.full_text):,} chars")
            print(f"   Sections detected: {len(sections)}")

            print(f"\n📑 Section Breakdown:")

            section_details = []
            for section_name, spans in sections.items():
                num_spans = len(spans)
                num_headings = sum(1 for s in spans if s.is_heading)
                num_text = num_spans - num_headings

                # Get sample text (first non-heading span)
                sample_text = ""
                for span in spans:
                    if not span.is_heading and len(span.text) > 50:
                        sample_text = span.text[:200] + "..." if len(span.text) > 200 else span.text
                        break

                section_details.append({
                    "name": section_name,
                    "total_spans": num_spans,
                    "headings": num_headings,
                    "text_spans": num_text,
                    "sample": sample_text
                })

                # Check if this is a risk-related section
                is_risk = any(keyword in section_name.lower() for keyword in
                             ["risk", "uncertainty", "challenge", "threat"])
                risk_marker = "🎯" if is_risk else "  "

                print(f"\n   {risk_marker} Section: {section_name}")
                print(f"      Spans: {num_spans:,} ({num_headings} headings, {num_text} text)")
                if sample_text:
                    print(f"      Sample: {sample_text[:100]}...")

            # Check for risk-related content in headings
            print(f"\n🔍 Risk-related headings found:")
            risk_headings = [
                span.text for span in extracted.spans
                if span.is_heading and any(
                    keyword in span.text.lower()
                    for keyword in ["risk", "uncertainty", "challenge", "threat", "principal"]
                )
            ]

            if risk_headings:
                for heading in risk_headings[:10]:  # Show first 10
                    print(f"      - {heading}")
            else:
                print(f"      ⚠️  No risk headings detected!")

            # Save detailed output
            output_dir = settings.output_dir / "diagnostic"
            output_dir.mkdir(exist_ok=True, parents=True)

            output_file = output_dir / f"{company_number}_{ticker}_sections.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Company: {company_name}\n")
                f.write(f"Format: {report_format}\n")
                f.write(f"File: {report_path.name}\n")
                f.write(f"\n{'='*80}\n")
                f.write(f"EXTRACTION SUMMARY\n")
                f.write(f"{'='*80}\n")
                f.write(f"Total spans: {len(extracted.spans):,}\n")
                f.write(f"Text length: {len(extracted.full_text):,} chars\n")
                f.write(f"Sections detected: {len(sections)}\n")
                f.write(f"\n{'='*80}\n")
                f.write(f"SECTIONS\n")
                f.write(f"{'='*80}\n\n")

                for detail in section_details:
                    f.write(f"Section: {detail['name']}\n")
                    f.write(f"  Total spans: {detail['total_spans']:,}\n")
                    f.write(f"  Headings: {detail['headings']}\n")
                    f.write(f"  Text spans: {detail['text_spans']}\n")
                    f.write(f"  Sample: {detail['sample']}\n\n")

                f.write(f"\n{'='*80}\n")
                f.write(f"ALL HEADINGS\n")
                f.write(f"{'='*80}\n\n")

                for i, span in enumerate(extracted.spans):
                    if span.is_heading:
                        f.write(f"{i+1}. {span.text}\n")

            print(f"\n💾 Detailed report saved: {output_file}")

            results.append({
                "company": company_name,
                "ticker": ticker,
                "format": report_format,
                "status": "success",
                "total_spans": len(extracted.spans),
                "sections": len(sections),
                "section_names": list(sections.keys()),
                "risk_headings": len(risk_headings)
            })

        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception(f"Error processing {company_name}")
            results.append({
                "company": company_name,
                "status": "error",
                "error": str(e)
            })

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    success_count = sum(1 for r in results if r.get("status") == "success")
    print(f"\nProcessed: {success_count}/{len(companies)} companies")

    # Section detection stats
    print("\n📊 Section Detection Statistics:")

    ixbrl_results = [r for r in results if r.get("format") == "ixbrl"]
    pdf_results = [r for r in results if r.get("format") == "pdf"]

    if ixbrl_results:
        avg_sections = sum(r.get("sections", 0) for r in ixbrl_results) / len(ixbrl_results)
        print(f"\n   iXBRL reports ({len(ixbrl_results)}):")
        print(f"      Average sections: {avg_sections:.1f}")
        risk_count = sum(1 for r in ixbrl_results if r.get("risk_headings", 0) > 0)
        print(f"      With risk headings: {risk_count}/{len(ixbrl_results)}")

    if pdf_results:
        avg_sections = sum(r.get("sections", 0) for r in pdf_results) / len(pdf_results)
        print(f"\n   PDF reports ({len(pdf_results)}):")
        print(f"      Average sections: {avg_sections:.1f}")
        risk_count = sum(1 for r in pdf_results if r.get("risk_headings", 0) > 0)
        print(f"      With risk headings: {risk_count}/{len(pdf_results)}")

    print(f"\n📁 Detailed reports saved to: {settings.results_dir / 'diagnostic'}/")
    print("\n✅ Diagnostic complete!")

    # Save summary JSON
    summary_file = settings.results_dir / "diagnostic" / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Summary JSON saved: {summary_file}")


if __name__ == "__main__":
    diagnose_all_reports()
