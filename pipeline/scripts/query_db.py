#!/usr/bin/env python3
"""Utility script to query the AIRO database."""

import sys
from pathlib import Path

# Add pipeline directory to path so we can import src as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database, Mention, Firm


def print_summary():
    """Print database summary statistics."""
    db = Database()
    session = db.get_session()

    try:
        # Count mentions
        mention_count = session.query(Mention).count()
        print(f"\n{'=' * 80}")
        print(f"AIRO DATABASE SUMMARY")
        print(f"{'=' * 80}")
        print(f"\nTotal Mentions: {mention_count}")

        # Count firms
        firm_count = session.query(Firm).count()
        print(f"Total Firms: {firm_count}")

        # List firms
        if firm_count > 0:
            print(f"\n{'=' * 80}")
            print("FIRMS PROCESSED")
            print(f"{'=' * 80}\n")

            firms = session.query(Firm).all()
            for firm in firms:
                print(f"{firm.firm_name} ({firm.report_year})")
                print(f"  - Total AI mentions: {firm.total_ai_mentions}")
                print(f"  - Risk mentions: {firm.total_ai_risk_mentions}")
                print(f"  - Dominant risk: {firm.dominant_tier_1_category or 'N/A'}")
                print(f"  - Governance: {firm.max_governance_maturity or 'None'}")
                print(f"  - Specificity ratio: {firm.specificity_ratio:.2f}")
                print(f"  - Mitigation gap: {firm.mitigation_gap_score:.2f}")
                print()

        # Show sample mentions
        if mention_count > 0:
            print(f"{'=' * 80}")
            print("SAMPLE MENTIONS (First 3)")
            print(f"{'=' * 80}\n")

            mentions = session.query(Mention).limit(3).all()
            for mention in mentions:
                print(f"ID: {mention.mention_id}")
                print(f"Firm: {mention.firm_name} ({mention.report_year})")
                print(f"Type: {mention.mention_type}")
                print(f"Risk: {mention.tier_1_category or 'N/A'}")
                print(f"Specificity: {mention.specificity_level}")
                print(f"Governance: {mention.governance_maturity or 'N/A'}")
                print(f"Confidence: {mention.confidence_score:.2f}")
                print(f"Text: {mention.text_excerpt[:200]}...")
                print(f"Reasoning: {mention.reasoning_summary}")
                print()

    finally:
        session.close()


def export_to_csv():
    """Export mentions to CSV for analysis."""
    import csv

    db = Database()
    session = db.get_session()

    try:
        mentions = session.query(Mention).all()

        output_file = Path("output/mentions_export.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'mention_id', 'firm_name', 'sector', 'report_year',
                'mention_type', 'tier_1_category', 'tier_2_driver',
                'specificity_level', 'governance_maturity', 'confidence_score',
                'text_excerpt'
            ])

            # Data
            for m in mentions:
                writer.writerow([
                    m.mention_id, m.firm_name, m.sector, m.report_year,
                    m.mention_type, m.tier_1_category, m.tier_2_driver,
                    m.specificity_level, m.governance_maturity,
                    m.confidence_score, m.text_excerpt
                ])

        print(f"\nExported {len(mentions)} mentions to {output_file}")

    finally:
        session.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query AIRO database")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export mentions to CSV"
    )

    args = parser.parse_args()

    if args.export:
        export_to_csv()
    else:
        print_summary()
