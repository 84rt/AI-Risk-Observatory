"""Phase 1 golden set helper: download, preprocess, and register documents.

This script:
- loads the golden set companies from `data/reference/golden_set_companies.csv` (or --companies)
- downloads 2024/2023 filings into `data/raw/{ixbrl,pdfs}/{year}/`
- writes ingestion metadata to `data/runs/{run_id}/ingestion.json`
- preprocesses the downloaded filings into `data/processed/{run_id}/documents.parquet`

Usage (examples):
    python scripts/golden_set_phase1.py --all
    python scripts/golden_set_phase1.py --download
    python scripts/golden_set_phase1.py --preprocess --run-id gs-phase1-20250101-120000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

# Ensure src is importable
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.companies_house import CompaniesHouseClient  # noqa: E402
from src.company_utils import load_companies_csv  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.database import Database  # noqa: E402
from src.identifiers import make_company_id, make_document_id  # noqa: E402
from src.ixbrl_extractor import iXBRLExtractor  # noqa: E402
from src.pdf_extractor import PDFExtractor  # noqa: E402
from src.preprocessor import Preprocessor, PreprocessingStrategy  # noqa: E402
from src.xbrl_filings_client import XBRLFilingsClient  # noqa: E402

settings = get_settings()

REFERENCE_CSV = settings.data_dir / "reference" / "golden_set_companies.csv"
RUNS_DIR = settings.data_dir / "runs"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def generate_run_id(prefix: str = "gs-phase1") -> str:
    """Create a run_id that is stable within a script invocation."""
    return f"{prefix}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"


def sha256_file(path: Path) -> str:
    """Return SHA256 checksum for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_run_dirs(run_id: str) -> Dict[str, Path]:
    """Create required directories for a run (append-only)."""
    raw_ixbrl = settings.raw_dir / "ixbrl"
    raw_pdfs = settings.raw_dir / "pdfs"
    run_dir = RUNS_DIR / run_id
    processed_dir = settings.processed_dir / run_id

    raw_ixbrl.mkdir(parents=True, exist_ok=True)
    raw_pdfs.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    return {
        "raw_ixbrl": raw_ixbrl,
        "raw_pdfs": raw_pdfs,
        "run_dir": run_dir,
        "processed_dir": processed_dir,
    }


def load_companies(path: Path) -> List[Dict[str, str]]:
    """Load companies from a reference CSV."""
    return load_companies_csv(path)


def save_json(obj: dict, path: Path) -> None:
    """Persist JSON with utf-8 and readable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------
def download_reports(
    companies: Iterable[Dict[str, str]],
    years: List[int],
    run_id: str,
) -> List[Dict]:
    """Download reports for the provided companies and years."""
    dirs = ensure_run_dirs(run_id)
    xbrl_client = XBRLFilingsClient()
    ch_client = CompaniesHouseClient()

    manifest: List[Dict] = []

    for company in companies:
        company_number = company.get("company_number")
        company_name = company["company_name"]
        lei = company.get("lei")
        sector = company.get("sector", "Unknown")
        company_id = company.get("company_id") or make_company_id(company_number, lei, company_name)
        ticker = company.get("ticker") or company_id

        for year in years:
            record = {
                "company_id": company_id,
                "company_number": company_number,
                "company_name": company_name,
                "ticker": ticker,
                "lei": lei,
                "cni_sector": sector,
                "year": int(year),
                "run_id": run_id,
                "status": "pending",
                "format": None,
                "raw_path": None,
                "checksum_sha256": None,
                "source": None,
                "error": None,
            }

            # Try iXBRL via filings.xbrl.org
            if lei:
                try:
                    ixbrl_dir = dirs["raw_ixbrl"] / str(year)
                    ixbrl_dir.mkdir(parents=True, exist_ok=True)
                    result = xbrl_client.fetch_annual_report(
                        lei=lei,
                        entity_name=company_name,
                        output_dir=ixbrl_dir,
                        year=year,
                    )
                    if result and result.get("path"):
                        path = Path(result["path"])
                        record.update(
                            {
                                "status": "downloaded",
                                "format": "ixbrl",
                                "raw_path": str(path),
                                "checksum_sha256": sha256_file(path),
                                "source": "filings.xbrl.org",
                            }
                        )
                        manifest.append(record)
                        print(f"[{company_name} {year}] ✅ iXBRL downloaded")
                        continue
                except Exception as exc:
                    record["error"] = f"ixbrl_failed:{exc}"
                    print(f"[{company_name} {year}] ⚠️ iXBRL failed: {exc}")

            # Fallback to Companies House PDF/iXBRL
            if company_number:
                try:
                    pdf_dir = dirs["raw_pdfs"] / str(year)
                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    result = ch_client.fetch_annual_report(
                        company_number=company_number,
                        company_name=company_name,
                        year=year,
                        output_dir=pdf_dir,
                    )
                    if result and result.get("path"):
                        path = Path(result["path"])
                        record.update(
                            {
                                "status": "downloaded",
                                "format": result.get("format", "pdf"),
                                "raw_path": str(path),
                                "checksum_sha256": sha256_file(path),
                                "source": "companies_house",
                            }
                        )
                        manifest.append(record)
                        print(f"[{company_name} {year}] ✅ Companies House downloaded")
                        continue
                except Exception as exc:
                    record["error"] = f"companies_house_failed:{exc}"
                    print(f"[{company_name} {year}] ❌ download failed: {exc}")
            else:
                record["error"] = "missing_company_number"
                print(f"[{company_name} {year}] ⚠️ missing company_number (Companies House skipped)")

            # If we reach here, download failed
            record["status"] = "missing"
            manifest.append(record)

    return manifest


def write_ingestion_manifest(run_id: str, manifest: List[Dict]) -> Path:
    """Persist ingestion manifest to /data/runs/{run_id}/ingestion.json."""
    run_dir = ensure_run_dirs(run_id)["run_dir"]
    manifest_path = run_dir / "ingestion.json"
    save_json({"run_id": run_id, "created_at": datetime.utcnow(), "records": manifest}, manifest_path)
    return manifest_path


def save_manifest_to_db(companies: List[Dict[str, str]], manifest: List[Dict]) -> None:
    """Save company list and manifest records into SQLite."""
    db = Database()
    session = db.get_session()
    try:
        for company in companies:
            db.upsert_company(session, company)

        for record in manifest:
            company_number = record.get("company_number")
            company_name = record.get("company_name") or ""
            lei = record.get("lei")
            sector = record.get("cni_sector") or record.get("sector") or "Unknown"
            year = record.get("year")
            fmt = record.get("format") or "unknown"

            company_id = record.get("company_id") or make_company_id(
                company_number, lei, company_name
            )
            document_id = make_document_id(
                company_number, lei, company_name, year, sector, fmt
            )

            db.upsert_document(
                session,
                {
                    "document_id": document_id,
                    "company_id": company_id,
                    "company_name": company_name,
                    "company_number": company_number,
                    "lei": lei,
                    "ticker": record.get("ticker"),
                    "sector": sector,
                    "report_year": year,
                    "source_format": record.get("format"),
                    "raw_path": record.get("raw_path"),
                    "checksum_sha256": record.get("checksum_sha256"),
                    "source": record.get("source"),
                    "status": record.get("status"),
                    "error": record.get("error"),
                    "run_id": record.get("run_id"),
                },
            )

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def load_manifest(run_id: str) -> List[Dict]:
    """Load ingestion manifest for a run_id."""
    manifest_path = ensure_run_dirs(run_id)["run_dir"] / "ingestion.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No ingestion manifest found at {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("records", [])


def preprocess_manifest(
    manifest: List[Dict],
    run_id: str,
    strategy: PreprocessingStrategy,
    include_context: bool,
) -> Path:
    """Preprocess downloaded files and write documents.parquet."""
    dirs = ensure_run_dirs(run_id)
    processed_dir = dirs["processed_dir"]

    ixbrl_extractor = iXBRLExtractor()
    pdf_extractor = PDFExtractor()
    preprocessor = Preprocessor(strategy=strategy, include_context=include_context)

    rows: List[Dict] = []
    for rec in manifest:
        if rec.get("status") != "downloaded" or not rec.get("raw_path"):
            continue

        raw_path = Path(rec["raw_path"])
        fmt = rec.get("format", "pdf")
        company_name = rec["company_name"]
        company_number = rec.get("company_number")
        year = rec["year"]
        sector = rec.get("cni_sector", "Unknown")

        try:
            if fmt == "ixbrl":
                extracted = ixbrl_extractor.extract_report(raw_path)
            else:
                extracted = pdf_extractor.extract_report(raw_path)

            preprocessed = preprocessor.process(extracted, firm_name=company_name)

            document_id = make_document_id(company_number, rec.get("lei"), company_name, year, sector, fmt)
            rows.append(
                {
                    "document_id": document_id,
                    "company_id": rec.get("company_id") or make_company_id(company_number, rec.get("lei"), company_name),
                    "company_number": company_number,
                    "company_name": company_name,
                    "ticker": rec.get("ticker"),
                    "lei": rec.get("lei"),
                    "cni_sector": sector,
                    "year": year,
                    "source_format": fmt,
                    "raw_path": str(raw_path),
                    "checksum_sha256": rec.get("checksum_sha256"),
                    "preprocess_strategy": strategy.value,
                    "spans_original": len(extracted.spans),
                    "spans_retained": preprocessed.metadata.get("filtered_spans"),
                    "sections_original": preprocessed.metadata.get("original_sections"),
                    "run_id": run_id,
                    "created_at": datetime.utcnow(),
                    "text_markdown": preprocessed.markdown_content,
                    "stats": preprocessed.stats,
                }
            )
            print(f"[{company_name} {year}] ✅ preprocessed ({fmt})")
        except Exception as exc:
            print(f"[{company_name} {year}] ❌ preprocess failed: {exc}")

    if not rows:
        raise RuntimeError("No documents were preprocessed; check downloads and inputs.")

    df = pd.DataFrame(rows)
    parquet_path = processed_dir / "documents.parquet"
    df.to_parquet(parquet_path, index=False)

    manifest_path = processed_dir / "documents_manifest.json"
    save_json({"run_id": run_id, "count": len(rows), "parquet": str(parquet_path)}, manifest_path)

    return parquet_path


def verify_run(run_id: str, years: Optional[List[int]] = None) -> None:
    """Sanity check that each CNI sector has coverage for the specified years."""
    companies = load_companies(REFERENCE_CSV)
    manifest = load_manifest(run_id)
    processed_path = settings.processed_dir / run_id / "documents.parquet"

    if years is None:
        years = sorted({rec["year"] for rec in manifest})

    processed_df = pd.read_parquet(processed_path) if processed_path.exists() else pd.DataFrame()

    print("\n" + "=" * 60)
    print(f"COVERAGE CHECK FOR RUN {run_id}")
    print("=" * 60)
    header = f"{'Sector':<20} {'Company':<32} {'Year':<6} {'Download':<10} {'Processed':<10}"
    print(header)
    print("-" * len(header))

    missing = []
    for company in companies:
        for year in years:
            downloaded = any(
                rec.get("company_id") == company.get("company_id")
                and rec.get("year") == year
                and rec.get("status") == "downloaded"
                for rec in manifest
            )
            processed = False
            if not processed_df.empty:
                processed = not processed_df[
                    (processed_df.company_id == company.get("company_id"))
                    & (processed_df.year == year)
                ].empty

            print(
                f"{company['sector']:<20} {company['company_name']:<32} {year:<6} "
                f"{'yes' if downloaded else 'no':<10} {'yes' if processed else 'no':<10}"
            )
            if not downloaded or not processed:
                missing.append((company["sector"], company["company_name"], year))

    if missing:
        print("\nMissing coverage for:")
        for sector, name, year in missing:
            print(f"- {sector} / {name} / {year}")
    else:
        print("\nAll requested company-year pairs have downloads and processed text.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Golden set Phase 1 helper")
    parser.add_argument("--download", action="store_true", help="Download filings only")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess downloaded filings")
    parser.add_argument("--all", action="store_true", help="Run download + preprocess")
    parser.add_argument("--verify", action="store_true", help="Check coverage for a run_id")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2024, 2023],
        help="Years to fetch (default: 2024 2023)",
    )
    parser.add_argument(
        "--companies",
        type=Path,
        default=REFERENCE_CSV,
        help=f"Path to companies CSV (default: {REFERENCE_CSV})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id to reuse; otherwise generated automatically",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="full",
        choices=[s.value for s in PreprocessingStrategy],
        help="Preprocessing strategy (default: full)",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include context around keyword matches (only relevant for keyword strategy)",
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Save companies + ingestion manifest into SQLite",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Default behavior (no flags) runs both download and preprocess
    do_download = args.download or args.all or (not args.download and not args.preprocess)
    do_preprocess = args.preprocess or args.all or (not args.download and not args.preprocess)
    run_id = args.run_id or generate_run_id()

    print(f"Run ID: {run_id}")

    # Download
    if do_download:
        companies = load_companies(args.companies)
        manifest = download_reports(companies, args.years, run_id=run_id)
        manifest_path = write_ingestion_manifest(run_id, manifest)
        print(f"Ingestion manifest written to {manifest_path}")
        if args.save_db:
            save_manifest_to_db(companies, manifest)
            print("Saved ingestion records to SQLite")

    # Preprocess
    if do_preprocess:
        manifest = load_manifest(run_id)
        parquet_path = preprocess_manifest(
            manifest=manifest,
            run_id=run_id,
            strategy=PreprocessingStrategy(args.strategy),
            include_context=args.include_context,
        )
        print(f"Processed documents written to {parquet_path}")

    if args.verify:
        verify_run(run_id, years=args.years)


if __name__ == "__main__":
    main()

