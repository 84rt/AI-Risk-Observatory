"""Load human golden set annotations into SQLite for parity with model outputs.

Reads annotations from `data/annotations/human/annotations.parquet` (or JSONL)
and inserts into `mentions` and `risk_classifications` tables. Skips existing
records to preserve append-only semantics.

Usage:
    python scripts/load_golden_to_db.py --annotations ../data/annotations/human/annotations.parquet
    python scripts/load_golden_to_db.py --run-id gs-phase1-20250101-120000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from src.config import get_settings
from src.database import Database, Mention, RiskClassification

settings = get_settings()


def load_annotations(path: Path) -> pd.DataFrame:
    """Load annotations from parquet or JSONL."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    # Fallback: pick parquet if exists next to given path
    parquet_candidate = path.with_suffix(".parquet")
    if parquet_candidate.exists():
        return pd.read_parquet(parquet_candidate)
    raise FileNotFoundError(f"Annotation file not found or unsupported format: {path}")


def build_risk_classifications(df: pd.DataFrame) -> List[RiskClassification]:
    """Aggregate mention-level annotations into document-level risk_classifications rows."""
    records: List[RiskClassification] = []
    grouped = df.groupby(["document_id", "company_number", "company_name", "cni_sector", "year"])

    for (_, company_number, company_name, sector, year), group in grouped:
        firm_id = group["ticker"].iloc[0] if "ticker" in group else company_number
        risk_labels = group.loc[
            group["dimension"].isin(["risk_disclosure", "harm", "ai_risk", "risk"]),
            "label",
        ].dropna().unique().tolist()

        evidence_map: Dict[str, List[str]] = defaultdict(list)
        confidence_map: Dict[str, float] = {}
        key_snippets: Dict[str, str] = {}

        for _, row in group.iterrows():
            dim = row["dimension"]
            excerpt = row.get("text_excerpt")
            if excerpt:
                evidence_map[dim].append(excerpt)
                key_snippets.setdefault(dim, excerpt)
            conf = row.get("confidence")
            if pd.notna(conf):
                confidence_map[dim] = max(confidence_map.get(dim, 0.0), float(conf))

        classification = RiskClassification(
            firm_id=str(firm_id),
            firm_name=company_name,
            company_number=str(company_number),
            sector=sector or "Unknown",
            report_year=int(year),
            ai_mentioned=not group.empty,
            risk_types=json.dumps(risk_labels),
            evidence=json.dumps(evidence_map),
            key_snippets=json.dumps(key_snippets),
            confidence_scores=json.dumps(confidence_map),
            reasoning="human_annotation",
            model_version="human_golden_v1",
            classification_date=date.today(),
            source_file="annotations/human/annotations.parquet",
        )
        records.append(classification)

    return records


def insert_annotations(
    df: pd.DataFrame,
    db: Database,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Insert annotations into mentions and risk_classifications tables."""
    session = db.get_session()
    counters = {"mentions_inserted": 0, "mentions_skipped": 0, "classifications_inserted": 0, "classifications_skipped": 0}

    try:
        # Mentions
        for _, row in df.iterrows():
            mention_id = row.get("annotation_id") or f"hum-{row['document_id']}-{row.name}"
            existing = session.get(Mention, mention_id)
            if existing:
                counters["mentions_skipped"] += 1
                continue

            mention = Mention(
                mention_id=mention_id,
                firm_id=str(row.get("ticker") or row.get("company_number")),
                firm_name=row.get("company_name"),
                sector=row.get("cni_sector") or "Unknown",
                sector_code=None,
                report_year=int(row.get("year")),
                report_section=row.get("report_section"),
                text_excerpt=row.get("text_excerpt"),
                page_number=None if pd.isna(row.get("page_number")) else int(row.get("page_number")),
                mention_type=row.get("dimension"),
                ai_specificity=row.get("ai_specificity") or "general",
                frontier_tech_flag=bool(row.get("frontier_tech_flag")) if not pd.isna(row.get("frontier_tech_flag")) else False,
                tier_1_category=row.get("tier_1_category") or row.get("label"),
                tier_2_driver=row.get("tier_2_driver"),
                specificity_level=row.get("specificity_level") or "general",
                materiality_signal=row.get("materiality_signal"),
                mitigation_mentioned=bool(row.get("mitigation_mentioned")) if not pd.isna(row.get("mitigation_mentioned")) else False,
                governance_maturity=row.get("governance_maturity"),
                confidence_score=float(row.get("confidence")) if pd.notna(row.get("confidence")) else None,
                reasoning_summary=row.get("notes"),
                model_version=row.get("classifier_version") or "human_golden_v1",
                extraction_date=date.today(),
                review_status="human-reviewed",
            )
            session.add(mention)
            counters["mentions_inserted"] += 1

        # Risk classifications (document-level)
        rc_records = build_risk_classifications(df)
        for rc in rc_records:
            existing_rc = session.get(RiskClassification, (rc.firm_id, rc.report_year))
            if existing_rc:
                counters["classifications_skipped"] += 1
                continue
            session.add(rc)
            counters["classifications_inserted"] += 1

        if dry_run:
            session.rollback()
        else:
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load golden set annotations into SQLite")
    parser.add_argument(
        "--annotations",
        type=str,
        default=str(settings.annotations_dir / "human" / "annotations.parquet"),
        help="Path to annotations parquet/jsonl",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id filter (only import rows matching run_id)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing to DB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ann_path = Path(args.annotations)
    df = load_annotations(ann_path)
    if args.run_id:
        df = df.loc[df["run_id"] == args.run_id]
        print(f"Filtered to run_id={args.run_id}: {len(df)} rows")
    else:
        print(f"Loaded {len(df)} annotations from {ann_path}")

    db = Database()
    counters = insert_annotations(df, db=db, dry_run=args.dry_run)
    print(json.dumps(counters, indent=2))
    if args.dry_run:
        print("Dry run complete (no changes committed).")


if __name__ == "__main__":
    main()

