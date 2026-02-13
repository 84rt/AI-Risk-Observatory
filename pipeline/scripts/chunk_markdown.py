#!/usr/bin/env python3
"""Chunk preprocessed markdown reports by AI keyword mentions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        help="Run ID for data/processed/<run_id>/documents_manifest.json",
    )
    parser.add_argument(
        "--context-before",
        type=int,
        default=2,
        help="Paragraphs of context before a match",
    )
    parser.add_argument(
        "--context-after",
        type=int,
        default=2,
        help="Paragraphs of context after a match",
    )
    parser.add_argument(
        "--max-chunk-words",
        type=int,
        default=600,
        help="Hard cap for chunk size in words; oversized chunks are split.",
    )
    parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=1,
        help="Sentence overlap between split subchunks of oversized chunks.",
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
    manifest_path = processed_dir / "documents_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing documents_manifest.json at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    documents = manifest.get("documents", [])
    if not documents:
        raise RuntimeError(f"No documents listed in {manifest_path}")

    all_chunks = []
    db = Database() if args.save_db else None
    session = db.get_session() if db else None

    try:
        for record in documents:
            document_id = record.get("document_id")
            company_id = record.get("company_id") or record.get("company_number") or document_id
            company_name = record.get("company_name") or company_id
            report_year = int(record.get("year"))
            markdown_path = Path(record["markdown_path"])
            if not markdown_path.exists():
                raise FileNotFoundError(f"Missing markdown file at {markdown_path}")
            markdown = markdown_path.read_text(encoding="utf-8")

            chunks = chunk_markdown(
                markdown=markdown,
                document_id=document_id,
                company_id=company_id,
                company_name=company_name,
                report_year=report_year,
                context_before=args.context_before,
                context_after=args.context_after,
                max_chunk_words=args.max_chunk_words,
                overlap_sentences=args.overlap_sentences,
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
                        "source_format": record.get("source_format"),
                        "preprocess_strategy": record.get("preprocess_strategy"),
                        "markdown_text": markdown,
                        "run_id": record.get("run_id") or args.run_id,
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

    output_dir = args.output_dir or (processed_dir / "chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"✅ Wrote {len(all_chunks)} chunks to {jsonl_path}")
    if args.save_db:
        print("✅ Saved processed documents + chunks to SQLite")


if __name__ == "__main__":
    main()
