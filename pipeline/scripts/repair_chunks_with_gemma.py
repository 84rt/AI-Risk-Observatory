#!/usr/bin/env python3
"""Repair chunk spacing issues using OpenRouter (Gemma 2 9b)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

# Ensure src is importable
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.config import get_settings  # noqa: E402

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Fix spacing and joining errors in the text. "
    "Do not change words, numbers, or facts. "
    "Return ONLY the corrected text."
)


class OpenRouterError(RuntimeError):
    """Raised for OpenRouter API errors."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"OpenRouter error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


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
    parser = argparse.ArgumentParser(description="Repair chunk text with OpenRouter")
    parser.add_argument("--run-id", required=True, help="Run ID for processed outputs")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-9b-it",
        help="OpenRouter model ID",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to chunks.jsonl (defaults to processed/<run_id>/chunks/chunks.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for repaired chunks (defaults to processed/<run_id>/chunks/chunks_gemma.jsonl)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output and skip already processed chunk_ids",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Stop after processing this many chunks (0 = no limit)",
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
    return parser.parse_args()


def load_processed_chunk_ids(path: Path) -> Set[str]:
    processed: Set[str] = set()
    if not path.exists():
        return processed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = record.get("chunk_id")
            if chunk_id:
                processed.add(chunk_id)
    return processed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    settings = get_settings()

    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    processed_dir = settings.processed_dir / args.run_id
    input_path = args.input or (processed_dir / "chunks" / "chunks.jsonl")
    output_path = args.output or (processed_dir / "chunks" / "chunks_gemma.jsonl")
    repairs_dir = processed_dir / "llm_repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    repairs_log = repairs_dir / "chunks_gemma_repairs.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing chunks at {input_path}")

    processed_ids = load_processed_chunk_ids(output_path) if args.append else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.append and repairs_log.exists() else "w"
    output_mode = "a" if args.append and output_path.exists() else "w"

    processed_count = 0
    with open(output_path, output_mode, encoding="utf-8") as out_f, open(
        repairs_log, log_mode, encoding="utf-8"
    ) as log_f:
        with open(input_path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                chunk: Dict[str, object]
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk_id = str(chunk.get("chunk_id", ""))
                if not chunk_id or chunk_id in processed_ids:
                    continue

                original_text = str(chunk.get("chunk_text", ""))
                if not original_text:
                    out_f.write(json.dumps(chunk) + "\n")
                    continue

                for attempt in range(1, args.max_retries + 1):
                    try:
                        corrected = openrouter_chat(
                            api_key=settings.openrouter_api_key,
                            base_url=settings.openrouter_base_url,
                            model=args.model,
                            prompt=original_text,
                            timeout=args.timeout,
                        )
                        chunk["chunk_text"] = corrected
                        out_f.write(json.dumps(chunk) + "\n")
                        log_f.write(json.dumps({
                            "chunk_id": chunk_id,
                            "accepted": True,
                            "original": original_text,
                            "corrected": corrected,
                        }) + "\n")
                        processed_count += 1
                        break
                    except OpenRouterError as exc:
                        if exc.status_code in {429, 500, 502, 503, 504}:
                            sleep_for = args.retry_sleep * attempt
                            logger.warning(
                                "Retrying after API error %s (sleep %.1fs)",
                                exc.status_code,
                                sleep_for,
                            )
                            time.sleep(sleep_for)
                            continue
                        logger.warning("OpenRouter error: %s", exc)
                        break
                    except Exception as exc:
                        logger.warning("OpenRouter call failed: %s", exc)
                        break

                if processed_count and processed_count % 50 == 0:
                    logger.info("Processed %d chunks", processed_count)

                if args.max_chunks > 0 and processed_count >= args.max_chunks:
                    logger.info("Reached max chunks; stopping early.")
                    return

    logger.info("Repaired chunks written to %s", output_path)
    logger.info("Repair log written to %s", repairs_log)
    logger.info("Total repaired chunks: %d", processed_count)


if __name__ == "__main__":
    main()
