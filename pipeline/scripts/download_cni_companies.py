"""Download annual reports for CNI sector companies."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import settings
from src.xbrl_filings_client import XBRLFilingsClient
from src.ixbrl_extractor import extract_text_from_ixbrl
from src.preprocessor import Preprocessor, PreprocessingStrategy

# Companies to download (5 new CNI sector companies)
NEW_CNI_COMPANIES = [
    {
        "lei": "2138001AVBSD1HSC6Z10",
        "company_name": "Johnson Matthey plc",
        "company_number": "00033774",
        "ticker": "JMAT",
        "cni_sector": "Chemicals"
    },
    {
        "lei": "213800LRO7NS5CYQMN21",
        "company_name": "BT Group plc",
        "company_number": "04190816",
        "ticker": "BT.A",
        "cni_sector": "Communications"
    },
    {
        "lei": "213800TSKOLX4EU6L377",
        "company_name": "Babcock International Group plc",
        "company_number": "02342138",
        "ticker": "BAB",
        "cni_sector": "Emergency Services"
    },
    {
        "lei": "CMIGEWPLHL4M7ZV0IZ88",
        "company_name": "Capita plc",
        "company_number": "02081330",
        "ticker": "CPI",
        "cni_sector": "Government"
    },
    {
        "lei": "213800RPBXRETY4A4C59",
        "company_name": "Severn Trent plc",
        "company_number": "02366619",
        "ticker": "SVT",
        "cni_sector": "Water"
    }
]

# Existing companies in golden set that we want for CNI coverage
EXISTING_CNI_COMPANIES = [
    {
        "company_name": "BAE Systems plc",
        "company_number": "01470151",
        "ticker": "BA",
        "cni_sector": "Defence"
    },
    {
        "company_name": "Shell plc",
        "company_number": "04366849",
        "ticker": "SHEL",
        "cni_sector": "Energy"
    },
    {
        "company_name": "HSBC Holdings plc",
        "company_number": "00617987",
        "ticker": "HSBA",
        "cni_sector": "Finance"
    },
    {
        "company_name": "Tesco PLC",
        "company_number": "00445790",
        "ticker": "TSCO",
        "cni_sector": "Food"
    },
    {
        "company_name": "AstraZeneca plc",
        "company_number": "02723534",
        "ticker": "AZN",
        "cni_sector": "Health"
    },
    {
        "company_name": "Rolls-Royce Holdings plc",
        "company_number": "07524813",
        "ticker": "RR",
        "cni_sector": "Civil Nuclear / Space"
    },
    {
        "company_name": "National Grid plc",
        "company_number": "04031152",
        "ticker": "NG",
        "cni_sector": "Transport"
    }
]


def download_company_reports(years=[2023, 2024]):
    """Download reports for new CNI companies."""
    client = XBRLFilingsClient()
    output_dir = settings.raw_dir / "ixbrl"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for company in NEW_CNI_COMPANIES:
        print(f"\n{'='*60}")
        print(f"Processing: {company['company_name']} ({company['cni_sector']})")
        print(f"LEI: {company['lei']}")
        print(f"{'='*60}")

        # Get all filings for this company
        filings = client.get_entity_filings(company['lei'], limit=20)

        if not filings:
            print(f"  ERROR: No filings found for {company['company_name']}")
            results.append({
                "company": company['company_name'],
                "status": "NO_FILINGS",
                "files": []
            })
            continue

        print(f"  Found {len(filings)} filings")

        # Show available years
        available_years = set()
        for f in filings:
            period_end = f.get("attributes", {}).get("period_end", "")
            if period_end:
                year = period_end[:4]
                available_years.add(year)
        print(f"  Available years: {sorted(available_years)}")

        downloaded = []
        for year in years:
            # Filter filings for this year
            year_filings = [
                f for f in filings
                if f.get("attributes", {}).get("period_end", "").startswith(str(year))
            ]

            if not year_filings:
                print(f"  No filing for year {year}")
                continue

            filing = year_filings[0]
            attrs = filing.get("attributes", {})
            period_end = attrs.get("period_end", "unknown")

            # Generate filename using company number
            filename = f"{company['company_number']}_{company['ticker']}_{year}.xhtml"
            output_path = output_dir / filename

            if output_path.exists():
                print(f"  {year}: Already exists - {output_path.name}")
                downloaded.append(str(output_path))
                continue

            try:
                client.download_xhtml_report(filing, output_path)
                print(f"  {year}: Downloaded - {output_path.name}")
                downloaded.append(str(output_path))
            except Exception as e:
                print(f"  {year}: ERROR - {e}")

        results.append({
            "company": company['company_name'],
            "company_number": company['company_number'],
            "cni_sector": company['cni_sector'],
            "status": "SUCCESS" if downloaded else "NO_DOWNLOADS",
            "files": downloaded
        })

    return results


def preprocess_downloaded_reports():
    """Preprocess downloaded reports using keyword strategy."""
    preprocessor = Preprocessor(strategy=PreprocessingStrategy.KEYWORD)
    output_dir = settings.processed_dir / "preprocessed" / "keyword"
    output_dir.mkdir(parents=True, exist_ok=True)

    ixbrl_dir = settings.raw_dir / "ixbrl"

    results = []

    for company in NEW_CNI_COMPANIES:
        print(f"\n{'='*60}")
        print(f"Preprocessing: {company['company_name']}")
        print(f"{'='*60}")

        # Find downloaded files for this company
        pattern = f"{company['company_number']}_{company['ticker']}_*.xhtml"
        files = list(ixbrl_dir.glob(pattern))

        if not files:
            print(f"  No files found matching {pattern}")
            continue

        for file_path in files:
            print(f"  Processing: {file_path.name}")

            try:
                # Extract text
                extracted = extract_text_from_ixbrl(file_path)

                # Get year from filename
                year = file_path.stem.split("_")[-1]

                # Preprocess
                preprocessed = preprocessor.process(extracted, firm_name=company['company_name'])

                # Generate output filename
                output_filename = f"{company['company_number']}_{company['ticker']}_{year}.md"
                output_path = output_dir / output_filename

                # Save
                preprocessor.save_to_file(preprocessed, output_path)

                print(f"    -> Saved to {output_path.name}")
                print(f"    -> Extracted {len(preprocessed.spans)} spans")

                results.append({
                    "company": company['company_name'],
                    "year": year,
                    "output_path": str(output_path),
                    "spans": len(preprocessed.spans)
                })

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    "company": company['company_name'],
                    "file": file_path.name,
                    "error": str(e)
                })

    return results


def verify_cni_coverage():
    """Verify we have all 13 CNI sectors covered."""
    keyword_dir = settings.processed_dir / "preprocessed" / "keyword"

    print("\n" + "="*60)
    print("CNI SECTOR COVERAGE VERIFICATION")
    print("="*60)

    all_companies = EXISTING_CNI_COMPANIES + [{
        "company_name": c["company_name"],
        "company_number": c["company_number"],
        "ticker": c["ticker"],
        "cni_sector": c["cni_sector"]
    } for c in NEW_CNI_COMPANIES]

    coverage = {}

    for company in all_companies:
        pattern = f"{company['company_number']}_{company['ticker']}_*.md"
        files = list(keyword_dir.glob(pattern))

        # Also try without ticker for existing files
        if not files:
            pattern2 = f"{company['company_number']}_*.md"
            files = list(keyword_dir.glob(pattern2))

        sector = company['cni_sector']
        if sector not in coverage:
            coverage[sector] = {"company": company['company_name'], "files": []}

        coverage[sector]["files"].extend([f.name for f in files])

    print(f"\n{'CNI Sector':<25} {'Company':<30} {'Files':<10}")
    print("-" * 70)

    missing = []
    for sector in sorted(coverage.keys()):
        data = coverage[sector]
        file_count = len(data["files"])
        status = "✓" if file_count > 0 else "✗"
        print(f"{sector:<25} {data['company']:<30} {file_count:<10} {status}")

        if file_count == 0:
            missing.append(sector)

    print("\n" + "-" * 70)
    if missing:
        print(f"MISSING: {', '.join(missing)}")
    else:
        print("All 13 CNI sectors covered!")

    return coverage


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CNI sector company reports")
    parser.add_argument("--download", action="store_true", help="Download reports")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess downloaded reports")
    parser.add_argument("--verify", action="store_true", help="Verify CNI coverage")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024], help="Years to download")

    args = parser.parse_args()

    if args.all or args.download:
        print("\n" + "="*60)
        print("STEP 1: DOWNLOADING REPORTS")
        print("="*60)
        download_results = download_company_reports(years=args.years)

        print("\n\nDOWNLOAD SUMMARY:")
        for r in download_results:
            print(f"  {r['company']}: {r['status']} ({len(r['files'])} files)")

    if args.all or args.preprocess:
        print("\n" + "="*60)
        print("STEP 2: PREPROCESSING REPORTS")
        print("="*60)
        preprocess_results = preprocess_downloaded_reports()

        print("\n\nPREPROCESSING SUMMARY:")
        for r in preprocess_results:
            if "error" in r:
                print(f"  {r['company']}: ERROR - {r['error']}")
            else:
                print(f"  {r['company']} ({r['year']}): {r['spans']} spans")

    if args.all or args.verify:
        verify_cni_coverage()

    if not (args.all or args.download or args.preprocess or args.verify):
        print("Usage: python download_cni_companies.py --download --preprocess --verify")
        print("   or: python download_cni_companies.py --all")
