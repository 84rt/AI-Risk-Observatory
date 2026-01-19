#!/usr/bin/env python3
"""Repair chunk spacing issues using OpenRouter (Gemma 2 9b) with concurrent processing."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiohttp

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


async def openrouter_chat(
    session: aiohttp.ClientSession,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
) -> str:
    """Make async request to OpenRouter API."""
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
    
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with session.post(url, headers=headers, json=payload, timeout=timeout_obj) as response:
        response_text = await response.text()
        if response.status != 200:
            raise OpenRouterError(response.status, response_text[:500])
        data = json.loads(response_text)
        return data["choices"][0]["message"]["content"].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair chunk text with OpenRouter (concurrent)")
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
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent requests (default: 50, can go higher with good credit balance)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for transient API errors (especially 429)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="OpenRouter request timeout seconds",
    )
    return parser.parse_args()


def load_processed_chunk_ids(path: Path) -> Set[str]:
    """Load already processed chunk IDs from output file."""
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
            if "chunk_id" not in record:
                continue
            chunk_id = record.get("chunk_id")
            if chunk_id is None:
                continue
            processed.add(str(chunk_id))
    return processed


def load_chunks_to_process(input_path: Path, processed_ids: Set[str], max_chunks: int) -> List[Dict]:
    """Load chunks that need processing."""
    chunks_to_process = []
    text_chunks_added = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            raw_chunk_id = chunk.get("chunk_id")
            if raw_chunk_id is None:
                continue
            chunk_id = str(raw_chunk_id)
            if not chunk_id or chunk_id in processed_ids:
                continue

            chunks_to_process.append(chunk)
            if chunk.get("chunk_text"):
                text_chunks_added += 1

            if max_chunks > 0 and text_chunks_added >= max_chunks:
                break
    
    return chunks_to_process


async def process_single_chunk(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    chunk: Dict,
    settings,
    args,
) -> Dict:
    """Process a single chunk with retry logic and exponential backoff."""
    chunk_id = chunk["chunk_id"]
    original_text = chunk["chunk_text"]

    if not original_text:
        return {
            "chunk": chunk,
            "success": True,
            "log": {
                "chunk_id": chunk_id,
                "accepted": False,
                "skipped": True,
                "reason": "empty_chunk_text",
            }
        }
    
    async with semaphore:  # Control concurrency
        for attempt in range(args.max_retries):
            try:
                corrected = await openrouter_chat(
                    session=session,
                    api_key=settings.openrouter_api_key,
                    base_url=settings.openrouter_base_url,
                    model=args.model,
                    prompt=original_text,
                    timeout=args.timeout,
                )
                
                return {
                    "chunk": {**chunk, "chunk_text": corrected},
                    "success": True,
                    "log": {
                        "chunk_id": chunk_id,
                        "accepted": True,
                        "original": original_text,
                        "corrected": corrected,
                    }
                }
                
            except OpenRouterError as exc:
                if exc.status_code == 429:
                    # Exponential backoff for rate limiting
                    wait_time = min(2 ** attempt, 32)
                    logger.warning(
                        "Rate limited (429) on chunk %s, attempt %d/%d. Waiting %ds",
                        chunk_id, attempt + 1, args.max_retries, wait_time
                    )
                    await asyncio.sleep(wait_time)
                    continue
                elif exc.status_code in {500, 502, 503, 504}:
                    # Server errors - retry with backoff
                    wait_time = min(2 ** attempt, 16)
                    logger.warning(
                        "Server error %s on chunk %s, attempt %d/%d. Waiting %ds",
                        exc.status_code, chunk_id, attempt + 1, args.max_retries, wait_time
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error("OpenRouter error on chunk %s: %s", chunk_id, exc)
                    return {
                        "chunk": chunk,
                        "success": False,
                        "error": str(exc),
                        "log": {
                            "chunk_id": chunk_id,
                            "accepted": False,
                            "error": str(exc),
                        }
                    }
                    
            except Exception as exc:
                logger.error("Unexpected error on chunk %s: %s", chunk_id, exc)
                return {
                    "chunk": chunk,
                    "success": False,
                    "error": str(exc),
                    "log": {
                        "chunk_id": chunk_id,
                        "accepted": False,
                        "error": str(exc),
                    }
                }
        
        # Max retries exceeded
        logger.error("Max retries exceeded for chunk %s", chunk_id)
        return {
            "chunk": chunk,
            "success": False,
            "error": "Max retries exceeded",
            "log": {
                "chunk_id": chunk_id,
                "accepted": False,
                "error": "Max retries exceeded",
            }
        }


async def process_chunks_concurrent(
    chunks: List[Dict],
    settings,
    args,
) -> List[Dict]:
    """Process all chunks concurrently with controlled parallelism."""
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    logger.info(
        "Processing %d chunks with max %d concurrent requests",
        len(chunks), args.max_concurrent
    )
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_single_chunk(session, semaphore, chunk, settings, args)
            for chunk in chunks
        ]
        
        # Process all chunks and maintain order
        results = await asyncio.gather(*tasks)
        
    return results


async def async_main() -> None:
    """Async main function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
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

    # Load already processed chunks if appending
    processed_ids = load_processed_chunk_ids(output_path) if args.append else set()
    logger.info("Already processed: %d chunks", len(processed_ids))
    
    # Load chunks to process
    chunks_to_process = load_chunks_to_process(input_path, processed_ids, args.max_chunks)
    
    if not chunks_to_process:
        logger.info("No chunks to process!")
        return
    
    logger.info("Starting concurrent processing of %d chunks", len(chunks_to_process))
    start_time = time.time()
    
    # Process all chunks concurrently
    results = await process_chunks_concurrent(chunks_to_process, settings, args)
    
    # Write results (maintaining order)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.append and repairs_log.exists() else "w"
    output_mode = "a" if args.append and output_path.exists() else "w"
    
    success_count = 0
    failure_count = 0
    
    with open(output_path, output_mode, encoding="utf-8") as out_f, \
         open(repairs_log, log_mode, encoding="utf-8") as log_f:
        
        for result in results:
            out_f.write(json.dumps(result["chunk"]) + "\n")
            log_f.write(json.dumps(result["log"]) + "\n")
            
            if result["success"]:
                success_count += 1
            else:
                failure_count += 1
    
    elapsed = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info("Repaired chunks written to: %s", output_path)
    logger.info("Repair log written to: %s", repairs_log)
    logger.info("Successfully processed: %d chunks", success_count)
    logger.info("Failed: %d chunks", failure_count)
    logger.info("Total time: %.1f seconds (%.2f chunks/sec)", elapsed, len(chunks_to_process) / elapsed)
    logger.info("=" * 60)


def main() -> None:
    """Entry point that runs async main."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
