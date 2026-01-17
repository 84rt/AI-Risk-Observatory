#!/usr/bin/env python3
"""Use an LLM to repair paragraph spacing for QA-flagged documents."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from src.config import get_settings
from src.ixbrl_extractor import _is_valid_word

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Act as a text restoration engine. Your sole task is to fix spacing and "
    "joining errors in the provided text. Do not summarize. Do not change facts "
    "or numbers. Maintain the original professional tone. Output ONLY the "
    "corrected text."
)


@dataclass
class Block:
    kind: str  # heading | paragraph | blank
    text: str
    paragraph_index: Optional[int] = None


def parse_blocks(markdown: str) -> List[Block]:
    blocks: List[Block] = []
    buffer: List[str] = []
    paragraph_index = 0

    def flush_paragraph():
        nonlocal paragraph_index
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(Block(kind="paragraph", text=text, paragraph_index=paragraph_index))
                paragraph_index += 1
        buffer.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush_paragraph()
            blocks.append(Block(kind="heading", text=line))
            continue
        if not stripped:
            flush_paragraph()
            blocks.append(Block(kind="blank", text=""))
            continue
        buffer.append(line)

    flush_paragraph()
    return blocks


def reassemble_blocks(blocks: List[Block]) -> str:
    lines = []
    for block in blocks:
        if block.kind == "heading":
            lines.append(block.text)
        elif block.kind == "blank":
            lines.append("")
        else:
            lines.append(block.text)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def needs_llm_repair(text: str) -> bool:
    if len(text) < 80:
        return False
    tokens = re.findall(r"[A-Za-z]{3,}", text)
    if not tokens:
        return False
    valid = sum(1 for t in tokens if _is_valid_word(t))
    valid_ratio = valid / max(len(tokens), 1)
    long_tokens = sum(1 for t in tokens if len(t) >= 12)
    long_ratio = long_tokens / max(len(tokens), 1)
    connector_runs = re.search(r"[a-z]{6,}(and|the|of|to)[a-z]{4,}", text)
    return valid_ratio < 0.6 or long_ratio > 0.15 or bool(connector_runs)


def openrouter_chat(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> str:
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair markdown paragraphs with OpenRouter")
    parser.add_argument("--run-id", required=True, help="Run ID for processed outputs")
    parser.add_argument(
        "--qa-report",
        type=Path,
        default=None,
        help="Optional path to qa_preprocessing_report.json",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="qa",
        choices=["qa", "heuristic", "all"],
        help="Paragraph selection mode: qa (flagged docs only), heuristic, or all",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-9b-it",
        help="OpenRouter model ID",
    )
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=30,
        help="Max paragraphs to repair per document",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for repaired markdown",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    settings = get_settings()

    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    processed_dir = settings.processed_dir / args.run_id
    manifest_path = processed_dir / "documents_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        documents = json.load(f).get("documents", [])

    flagged_ids = set()
    if args.mode == "qa":
        qa_report = args.qa_report or (processed_dir / "qa_preprocessing_report.json")
        if not qa_report.exists():
            raise FileNotFoundError(f"Missing QA report at {qa_report}")

        with open(qa_report, "r", encoding="utf-8") as f:
            qa_results = json.load(f)

        flagged_ids = {r["document_id"] for r in qa_results if r.get("issues")}
        if not flagged_ids:
            logger.info("No QA-flagged documents found; nothing to repair.")
            return

    output_dir = args.output_dir or (processed_dir / "documents_repaired")
    output_dir.mkdir(parents=True, exist_ok=True)
    repairs_dir = processed_dir / "llm_repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    repairs_log = repairs_dir / "repairs.jsonl"

    repaired_count = 0
    with open(repairs_log, "w", encoding="utf-8") as log_f:
        for doc in documents:
            if args.mode == "qa" and doc.get("document_id") not in flagged_ids:
                continue

            markdown_path = Path(doc["markdown_path"])
            markdown = markdown_path.read_text(encoding="utf-8")
            blocks = parse_blocks(markdown)

            repair_targets: List[Tuple[int, Block]] = []
            for idx, block in enumerate(blocks):
                if block.kind != "paragraph":
                    continue
                if args.mode == "all":
                    repair_targets.append((idx, block))
                elif args.mode == "heuristic":
                    if needs_llm_repair(block.text):
                        repair_targets.append((idx, block))
                else:
                    if needs_llm_repair(block.text):
                        repair_targets.append((idx, block))

            if not repair_targets:
                continue

            for idx, block in repair_targets[: args.max_paragraphs]:
                original = block.text
                try:
                    corrected = openrouter_chat(
                        api_key=settings.openrouter_api_key,
                        base_url=settings.openrouter_base_url,
                        model=args.model,
                        prompt=original,
                    )
                except Exception as exc:
                    logger.warning("LLM call failed (%s): %s", doc.get("document_id"), exc)
                    continue

                accepted = digits_only(original) == digits_only(corrected)
                if accepted:
                    blocks[idx].text = corrected
                    repaired_count += 1

                log_f.write(json.dumps({
                    "document_id": doc.get("document_id"),
                    "paragraph_index": block.paragraph_index,
                    "accepted": accepted,
                    "original": original,
                    "corrected": corrected,
                }) + "\n")

            repaired_markdown = reassemble_blocks(blocks)
            repaired_path = output_dir / f"{doc.get('document_id')}.md"
            repaired_path.write_text(repaired_markdown, encoding="utf-8")

    logger.info("Repaired paragraphs written to %s", output_dir)
    logger.info("Repair log written to %s", repairs_log)
    logger.info("Accepted repairs: %d", repaired_count)


if __name__ == "__main__":
    main()
