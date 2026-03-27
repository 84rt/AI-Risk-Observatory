#!/usr/bin/env python3
"""Chunk preprocessed markdown reports by AI keyword mentions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable, desc="", unit="item"):
            self._iterable = list(iterable)
            self.desc = desc or "Progress"
            self.unit = unit
            self.total = len(self._iterable)
            self._index = 0
            self._postfix = ""
            self._last_emit = 0.0

        def __iter__(self):
            for item in self._iterable:
                yield item
                self._index += 1
                now = time.time()
                should_emit = (
                    self._index == 1
                    or self._index == self.total
                    or self._index % 10 == 0
                    or (now - self._last_emit) >= 2.0
                )
                if should_emit:
                    self._last_emit = now
                    pct = (self._index / self.total * 100) if self.total else 100.0
                    suffix = f" | {self._postfix}" if self._postfix else ""
                    print(
                        f"{self.desc}: {self._index}/{self.total} {self.unit}s ({pct:.1f}%)"
                        f"{suffix}",
                        flush=True,
                    )

        def set_postfix_str(self, value: str) -> None:
            self._postfix = value

from src.config import get_settings
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
        default=0,
        help="Sentence overlap between split subchunks of oversized chunks.",
    )
    parser.add_argument(
        "--keep-table-rule-lines",
        action="store_true",
        help="Keep markdown table separator lines (default is to drop them).",
    )
    parser.add_argument(
        "--keep-listing-signature-rows",
        action="store_true",
        help="Keep long listing/register table rows (default is to drop them).",
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
    db = None
    if args.save_db:
        from src.database import Database

        db = Database()
    session = db.get_session() if db else None

    try:
        progress = tqdm(documents, desc="Chunking reports", unit="report")
        for record in progress:
            document_id = record.get("document_id")
            company_id = record.get("company_id") or record.get("company_number") or document_id
            company_name = record.get("company_name") or company_id
            report_year = int(record.get("year"))
            market_segment = record.get("market_segment", "Other")
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
                drop_table_rule_lines=not args.keep_table_rule_lines,
                drop_listing_rows=not args.keep_listing_signature_rows,
            )
            for chunk in chunks:
                chunk["market_segment"] = market_segment
            all_chunks.extend(chunks)
            progress.set_postfix_str(
                f"{company_name[:32]} FY{report_year} chunks={len(chunks)} total={len(all_chunks)}"
            )

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
