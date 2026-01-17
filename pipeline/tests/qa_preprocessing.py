#!/usr/bin/env python3
"""Quality checks for preprocessed markdown outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.config import get_settings  # noqa: E402
from src.ixbrl_extractor import SECTION_PATTERNS  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="QA checks for preprocessed markdown")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID for data/processed/<run_id>/documents_manifest.json",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20000,
        help="Minimum character count for markdown",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3000,
        help="Minimum word count for markdown",
    )
    parser.add_argument(
        "--min-headings",
        type=int,
        default=8,
        help="Minimum number of markdown headings",
    )
    parser.add_argument(
        "--max-control-char-ratio",
        type=float,
        default=0.0001,
        help="Max ratio of control characters (excluding \\n/\\r/\\t)",
    )
    parser.add_argument(
        "--max-replacement-char-count",
        type=int,
        default=5,
        help="Max number of Unicode replacement chars (\\uFFFD)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for QA reports",
    )
    return parser.parse_args()


def normalize_heading(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"[^\w\s]", "", text).lower()


def extract_headings(markdown: str) -> List[str]:
    headings = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            heading_text = line.lstrip("#").strip()
            if heading_text:
                headings.append(heading_text)
    return headings


def section_coverage(headings: List[str]) -> Tuple[List[str], List[str]]:
    patterns = [re.compile(pat, re.IGNORECASE) for pat in SECTION_PATTERNS]
    normalized = [normalize_heading(h) for h in headings]

    found = []
    missing = []
    for pat in patterns:
        matched = any(pat.search(h) for h in normalized)
        if matched:
            found.append(pat.pattern)
        else:
            missing.append(pat.pattern)
    return found, missing


def sentence_stats(text: str) -> Dict[str, float]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    lengths = [len(s) for s in sentences if len(s) > 0]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    short_sentences = sum(1 for s in lengths if s < 20)
    long_sentences = sum(1 for s in lengths if s > 300)
    return {
        "sentence_count": len(sentences),
        "avg_sentence_length": avg_len,
        "short_sentence_ratio": (short_sentences / len(lengths)) if lengths else 0.0,
        "long_sentence_ratio": (long_sentences / len(lengths)) if lengths else 0.0,
    }


def control_char_stats(text: str) -> Dict[str, int]:
    control_chars = [
        ch for ch in text
        if (ord(ch) < 32 and ch not in "\n\r\t") or (0x7F <= ord(ch) <= 0x9F)
    ]
    replacement_chars = text.count("\uFFFD")
    return {
        "control_char_count": len(control_chars),
        "replacement_char_count": replacement_chars,
    }


def qa_document(record: Dict, markdown: str, thresholds: dict) -> Dict:
    headings = extract_headings(markdown)
    found_sections, missing_sections = section_coverage(headings)

    words = re.findall(r"\b\w+\b", markdown)
    char_count = len(markdown)
    word_count = len(words)

    sentence_info = sentence_stats(markdown)
    control_info = control_char_stats(markdown)

    control_ratio = control_info["control_char_count"] / max(len(markdown), 1)

    issues = []
    if char_count < thresholds["min_chars"]:
        issues.append("too_short_chars")
    if word_count < thresholds["min_words"]:
        issues.append("too_short_words")
    if len(headings) < thresholds["min_headings"]:
        issues.append("few_headings")
    if control_ratio > thresholds["max_control_char_ratio"]:
        issues.append("control_chars")
    if control_info["replacement_char_count"] > thresholds["max_replacement_char_count"]:
        issues.append("replacement_chars")
    if sentence_info["sentence_count"] == 0:
        issues.append("no_sentences")
    if sentence_info["long_sentence_ratio"] > 0.2:
        issues.append("many_long_sentences")

    return {
        "document_id": record.get("document_id"),
        "company_name": record.get("company_name"),
        "company_number": record.get("company_number"),
        "lei": record.get("lei"),
        "year": record.get("year"),
        "source_format": record.get("source_format"),
        "char_count": char_count,
        "word_count": word_count,
        "heading_count": len(headings),
        "sections_found": found_sections,
        "sections_missing": missing_sections,
        "sentence_count": sentence_info["sentence_count"],
        "avg_sentence_length": sentence_info["avg_sentence_length"],
        "short_sentence_ratio": sentence_info["short_sentence_ratio"],
        "long_sentence_ratio": sentence_info["long_sentence_ratio"],
        "control_char_count": control_info["control_char_count"],
        "control_char_ratio": control_ratio,
        "replacement_char_count": control_info["replacement_char_count"],
        "issues": issues,
    }


def run_qa(
    run_id: str,
    output_dir: Optional[Path] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    settings = get_settings()
    processed_dir = settings.processed_dir / run_id
    manifest_path = processed_dir / "documents_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing documents_manifest.json at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    documents = manifest.get("documents", [])
    if not documents:
        raise RuntimeError(f"No documents listed in {manifest_path}")

    thresholds = thresholds or {
        "min_chars": 20000,
        "min_words": 3000,
        "min_headings": 8,
        "max_control_char_ratio": 0.0001,
        "max_replacement_char_count": 5,
    }

    results = []
    for record in documents:
        markdown_path = Path(record.get("markdown_path", ""))
        if not markdown_path.exists():
            raise FileNotFoundError(f"Missing markdown file at {markdown_path}")
        markdown = markdown_path.read_text(encoding="utf-8")
        results.append(qa_document(record, markdown, thresholds))

    output_dir = output_dir or processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "qa_preprocessing_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_rows = []
    for result in results:
        csv_rows.append(
            {
                **{k: v for k, v in result.items() if k not in ["sections_found", "sections_missing", "issues"]},
                "sections_missing": ";".join(result["sections_missing"]),
                "issues": ";".join(result["issues"]),
            }
        )
    csv_path = output_dir / "qa_preprocessing_report.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    flagged = sum(1 for r in results if r["issues"])
    print(f"✅ QA complete: {len(results)} docs checked, {flagged} flagged")
    print(f"Report: {json_path}")
    print(f"Report: {csv_path}")
    return results


def main() -> None:
    args = parse_args()
    thresholds = {
        "min_chars": args.min_chars,
        "min_words": args.min_words,
        "min_headings": args.min_headings,
        "max_control_char_ratio": args.max_control_char_ratio,
        "max_replacement_char_count": args.max_replacement_char_count,
    }
    run_qa(
        run_id=args.run_id,
        output_dir=args.output_dir,
        thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
