#!/usr/bin/env python3
"""Chunk preprocessed markdown reports by AI keyword mentions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.database import Database
from src.markdown_chunker import chunk_markdown


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Chunk markdown reports by AI keywords")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID for data/processed/<run_id>/documents.parquet",
    )
    parser.add_argument(
        "--context-before",
        type=int,
        default=1,
        help="Paragraphs of context before a match",
    )
    parser.add_argument(
        "--context-after",
        type=int,
        default=1,
        help="Paragraphs of context after a match",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for chunk exports",
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Save processed documents and chunks into SQLite",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    processed_dir = settings.processed_dir / args.run_id
    parquet_path = processed_dir / "documents.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing documents.parquet at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise RuntimeError(f"No rows found in {parquet_path}")

    all_chunks = []
    db = Database() if args.save_db else None
    session = db.get_session() if db else None

    try:
        for _, row in df.iterrows():
            document_id = row.get("document_id")
            company_id = row.get("company_id") or row.get("company_number") or document_id
            company_name = row.get("company_name") or company_id
            report_year = int(row.get("year"))
            markdown = row.get("text_markdown") or ""

            chunks = chunk_markdown(
                markdown=markdown,
                document_id=document_id,
                company_id=company_id,
                company_name=company_name,
                report_year=report_year,
                context_before=args.context_before,
                context_after=args.context_after,
            )
            all_chunks.extend(chunks)

            if db:
                db.upsert_processed_document(
                    session,
                    {
                        "processed_id": document_id,
                        "document_id": document_id,
                        "company_id": company_id,
                        "company_name": company_name,
                        "report_year": report_year,
                        "source_format": row.get("source_format"),
                        "preprocess_strategy": row.get("preprocess_strategy"),
                        "markdown_text": markdown,
                        "run_id": row.get("run_id") or args.run_id,
                    },
                )

                for chunk in chunks:
                    db.upsert_document_chunk(
                        session,
                        {
                            "processed_id": document_id,
                            **chunk,
                        },
                    )

        if session:
            session.commit()
    except Exception:
        if session:
            session.rollback()
        raise
    finally:
        if session:
            session.close()

    output_dir = args.output_dir or processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.parquet"
    pd.DataFrame(all_chunks).to_parquet(chunks_path, index=False)

    jsonl_path = output_dir / "chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"✅ Wrote {len(all_chunks)} chunks to {chunks_path}")
    print(f"✅ Wrote {len(all_chunks)} chunks to {jsonl_path}")
    if args.save_db:
        print("✅ Saved processed documents + chunks to SQLite")


if __name__ == "__main__":
    main()
