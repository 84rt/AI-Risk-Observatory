#!/usr/bin/env python3
"""Repair markdown spacing issues using OpenRouter (Gemma 2 9b)."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# Ensure src is importable
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.config import get_settings  # noqa: E402
from src.ixbrl_extractor import _is_valid_word  # noqa: E402

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a text restoration engine. Fix spacing and joining errors only. "
    "Do not change words, numbers, or facts. Preserve paragraph breaks. "
    "Return ONLY the corrected text."
)


class OpenRouterError(RuntimeError):
    """Raised for OpenRouter API errors."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"OpenRouter error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class MaxCallsReached(RuntimeError):
    """Raised when hitting the max OpenRouter call limit."""


@dataclass
class Block:
    kind: str  # heading | paragraph | blank
    text: str
    paragraph_index: Optional[int] = None


def parse_blocks(markdown: str) -> List[Block]:
    blocks: List[Block] = []
    buffer: List[str] = []
    paragraph_index = 0

    def flush_paragraph() -> None:
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
    lines: List[str] = []
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


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", text)


def openrouter_chat(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
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
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise OpenRouterError(response.status_code, response.text[:500])
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair markdown spacing with OpenRouter")
    parser.add_argument("--run-id", required=True, help="Run ID for processed outputs")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "heuristic"],
        help="Paragraph selection mode: all or heuristic",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-9b-it",
        help="OpenRouter model ID",
    )
    parser.add_argument(
        "--max-paragraphs-per-request",
        type=int,
        default=8,
        help="Max paragraphs per OpenRouter call",
    )
    parser.add_argument(
        "--max-chars-per-request",
        type=int,
        default=12000,
        help="Target max chars per OpenRouter call",
    )
    parser.add_argument(
        "--min-chars-per-request",
        type=int,
        default=2000,
        help="Minimum chars before splitting further",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for transient API errors",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=2.0,
        help="Base sleep seconds for retries/backoff",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="OpenRouter request timeout seconds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for repaired markdown",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Stop after processing this many documents (0 = no limit)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="Stop after this many OpenRouter calls (0 = no limit)",
    )
    return parser.parse_args()


def split_paragraphs(
    items: List[Tuple[int, Block]],
    max_chars: int,
    max_paragraphs: int,
) -> List[List[Tuple[int, Block]]]:
    batches: List[List[Tuple[int, Block]]] = []
    current: List[Tuple[int, Block]] = []
    current_chars = 0

    for item in items:
        text = item[1].text
        item_chars = len(text) + 2
        if current and (
            current_chars + item_chars > max_chars or len(current) >= max_paragraphs
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars

    if current:
        batches.append(current)
    return batches


def split_texts(text: str) -> List[str]:
    return [t.strip() for t in re.split(r"\n\s*\n", text.strip()) if t.strip()]


def repair_batch(
    items: List[Tuple[int, Block]],
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
    max_calls: int,
    call_counter: List[int],
) -> Optional[List[str]]:
    prompt = "\n\n".join(block.text for _, block in items)
    if max_calls > 0 and call_counter[0] >= max_calls:
        raise MaxCallsReached("Max OpenRouter call limit reached.")
    call_counter[0] += 1
    for attempt in range(1, max_retries + 1):
        try:
            corrected = openrouter_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout=timeout,
            )
            return split_texts(corrected)
        except OpenRouterError as exc:
            if exc.status_code in {429, 500, 502, 503, 504}:
                sleep_for = retry_sleep * attempt
                logger.warning("Retrying after API error %s (sleep %.1fs)", exc.status_code, sleep_for)
                time.sleep(sleep_for)
                continue
            logger.warning("OpenRouter error: %s", exc)
            return None
        except Exception as exc:
            logger.warning("OpenRouter call failed: %s", exc)
            return None
    return None


def repair_items_recursive(
    items: List[Tuple[int, Block]],
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
    min_chars: int,
    max_calls: int,
    call_counter: List[int],
) -> List[Tuple[int, str, bool]]:
    if not items:
        return []

    combined_len = sum(len(block.text) for _, block in items)
    corrected_paragraphs = repair_batch(
        items=items,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        max_calls=max_calls,
        call_counter=call_counter,
    )

    if corrected_paragraphs and len(corrected_paragraphs) == len(items):
        results = []
        for (idx, block), corrected in zip(items, corrected_paragraphs):
            accepted = digits_only(block.text) == digits_only(corrected)
            results.append((idx, corrected if accepted else block.text, accepted))
        return results

    if len(items) == 1 or combined_len <= min_chars:
        idx, block = items[0]
        return [(idx, block.text, False)]

    mid = len(items) // 2
    left = repair_items_recursive(
        items=items[:mid],
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        min_chars=min_chars,
        max_calls=max_calls,
        call_counter=call_counter,
    )
    right = repair_items_recursive(
        items=items[mid:],
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        min_chars=min_chars,
        max_calls=max_calls,
        call_counter=call_counter,
    )
    return left + right


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

    output_dir = args.output_dir or (processed_dir / "documents_gemma_repaired")
    output_dir.mkdir(parents=True, exist_ok=True)
    repairs_dir = processed_dir / "llm_repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    repairs_log = repairs_dir / "gemma_repairs.jsonl"

    repaired_count = 0
    call_counter = [0]
    processed_docs = 0
    with open(repairs_log, "w", encoding="utf-8") as log_f:
        for doc in documents:
            markdown_path = Path(doc["markdown_path"])
            markdown = markdown_path.read_text(encoding="utf-8")
            blocks = parse_blocks(markdown)

            candidates: List[Tuple[int, Block]] = []
            for idx, block in enumerate(blocks):
                if block.kind != "paragraph":
                    continue
                if args.mode == "all" or needs_llm_repair(block.text):
                    candidates.append((idx, block))

            if not candidates:
                continue

            batches = split_paragraphs(
                items=candidates,
                max_chars=args.max_chars_per_request,
                max_paragraphs=args.max_paragraphs_per_request,
            )

            for batch in batches:
                try:
                    results = repair_items_recursive(
                        items=batch,
                        api_key=settings.openrouter_api_key,
                        base_url=settings.openrouter_base_url,
                        model=args.model,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        retry_sleep=args.retry_sleep,
                        min_chars=args.min_chars_per_request,
                        max_calls=args.max_calls,
                        call_counter=call_counter,
                    )
                except MaxCallsReached:
                    logger.info("Reached max calls; stopping early.")
                    results = []
                    break
                for idx, corrected, accepted in results:
                    original = blocks[idx].text
                    if accepted and corrected != original:
                        blocks[idx].text = corrected
                        repaired_count += 1

                    log_f.write(json.dumps({
                        "document_id": doc.get("document_id"),
                        "paragraph_index": blocks[idx].paragraph_index,
                        "accepted": accepted,
                        "original": original,
                        "corrected": corrected,
                    }) + "\n")

            repaired_markdown = reassemble_blocks(blocks)
            repaired_path = output_dir / f"{doc.get('document_id')}.md"
            repaired_path.write_text(repaired_markdown, encoding="utf-8")
            processed_docs += 1
            logger.info(
                "Processed %s (%d calls so far)",
                doc.get("document_id"),
                call_counter[0],
            )
            if args.max_docs > 0 and processed_docs >= args.max_docs:
                break

    logger.info("Repaired paragraphs written to %s", output_dir)
    logger.info("Repair log written to %s", repairs_log)
    logger.info("Accepted repairs: %d", repaired_count)


if __name__ == "__main__":
    main()
