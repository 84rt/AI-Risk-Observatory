#!/usr/bin/env python3
"""Parallel wrapper around chunk_markdown.py.

Splits the documents_manifest.json into N shards, runs chunk_markdown.py
on each shard in parallel, then merges all chunks.jsonl files into one.

Usage:
    python scripts/chunk_parallel.py --run-id unified-20260330 [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
CHUNK_SCRIPT = SCRIPT_DIR / "chunk_markdown.py"
DATA_ROOT = PIPELINE_ROOT.parent / "data"
PROCESSED_DIR = DATA_ROOT / "processed"

# Use the venv Python if it exists, otherwise fall back to sys.executable
_VENV_PYTHON = PIPELINE_ROOT / "venv" / "bin" / "python3"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel chunking across all documents.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel processes (default: 8)")
    parser.add_argument("--context-before", type=int, default=1)
    parser.add_argument("--context-after", type=int, default=1)
    parser.add_argument("--max-chunk-words", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = PROCESSED_DIR / args.run_id
    manifest_path = run_dir / "documents_manifest.json"

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = manifest["documents"]
    total = len(docs)
    n = args.workers

    print(f"Documents: {total}  Workers: {n}  (~{total // n} docs/worker)")

    # Split into N shards and write temporary shard manifests
    shard_size = (total + n - 1) // n
    shards = [docs[i:i + shard_size] for i in range(0, total, shard_size)]

    tmpdir = Path(tempfile.mkdtemp(prefix="chunk_shards_"))
    shard_run_ids = []
    shard_dirs = []

    for idx, shard_docs in enumerate(shards):
        shard_run_id = f"{args.run_id}_shard{idx:02d}"
        shard_run_dir = PROCESSED_DIR / shard_run_id
        shard_run_dir.mkdir(parents=True, exist_ok=True)

        shard_manifest = dict(manifest)
        shard_manifest["documents"] = shard_docs
        shard_manifest["document_count"] = len(shard_docs)
        (shard_run_dir / "documents_manifest.json").write_text(
            json.dumps(shard_manifest, ensure_ascii=False), encoding="utf-8"
        )
        shard_run_ids.append(shard_run_id)
        shard_dirs.append(shard_run_dir)

    print(f"Shards created in {PROCESSED_DIR}")
    print("Launching workers…")

    procs = []
    for shard_run_id in shard_run_ids:
        cmd = [
            PYTHON, str(CHUNK_SCRIPT),
            "--run-id", shard_run_id,
            "--context-before", str(args.context_before),
            "--context-after", str(args.context_after),
            "--max-chunk-words", str(args.max_chunk_words),
        ]
        log_path = PROCESSED_DIR / shard_run_id / "chunk.log"
        log_file = log_path.open("w")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, cwd=PIPELINE_ROOT)
        procs.append((shard_run_id, proc, log_file))
        print(f"  Started {shard_run_id} (pid={proc.pid})")

    # Wait for all workers
    print("\nWaiting for workers to finish…")

    import re
    import threading
    import time

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    # Track per-shard progress by tailing log files
    shard_progress = {rid: 0 for rid, _, _ in procs}
    shard_totals = {rid: shard_size for rid, _, _ in procs}
    done_flags = {rid: False for rid, _, _ in procs}
    _lock = threading.Lock()

    def _poll_logs() -> None:
        log_paths = {rid: PROCESSED_DIR / rid / "chunk.log" for rid, _, _ in procs}
        progress_re = re.compile(r"(\d+)/(\d+)")
        while not all(done_flags.values()):
            for rid, log_path in log_paths.items():
                if done_flags[rid] or not log_path.exists():
                    continue
                try:
                    # Read last 2KB to find the most recent progress line
                    with log_path.open("rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 2048))
                        tail = f.read().decode("utf-8", errors="replace")
                    for line in reversed(tail.splitlines()):
                        m = progress_re.search(line)
                        if m:
                            with _lock:
                                shard_progress[rid] = int(m.group(1))
                            break
                except OSError:
                    pass
            time.sleep(1.5)

    poll_thread = threading.Thread(target=_poll_logs, daemon=True)
    poll_thread.start()

    if _tqdm:
        bar = _tqdm(total=total, desc="Chunking", unit="doc", smoothing=0.1)
        last_total = 0

    failed = []
    for shard_run_id, proc, log_file in procs:
        # Poll until this shard finishes, updating tqdm from all shards
        while proc.poll() is None:
            if _tqdm:
                with _lock:
                    current = sum(shard_progress.values())
                delta = current - last_total
                if delta > 0:
                    bar.update(delta)
                    last_total = current
            time.sleep(1.5)

        log_file.close()
        done_flags[shard_run_id] = True
        ret = proc.returncode

        if _tqdm:
            # Snap this shard to its full size
            with _lock:
                shard_progress[shard_run_id] = shard_totals[shard_run_id]
            current = sum(shard_progress.values())
            bar.update(current - last_total)
            last_total = current

        status = "done" if ret == 0 else f"FAILED (exit {ret})"
        chunk_file = PROCESSED_DIR / shard_run_id / "chunks" / "chunks.jsonl"
        n_chunks = sum(1 for _ in chunk_file.open()) if chunk_file.exists() else 0
        if not _tqdm:
            print(f"  {shard_run_id}: {status}  chunks={n_chunks}")
        if ret != 0:
            failed.append(shard_run_id)

    if _tqdm:
        bar.update(total - last_total)  # ensure bar reaches 100%
        bar.close()

    done_flags.update({rid: True for rid in done_flags})

    if failed:
        print(f"\nFailed shards: {failed}", file=sys.stderr)
        return 1

    # Merge all chunks.jsonl into the main run dir
    print("\nMerging chunks…")
    output_dir = run_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "chunks.jsonl"

    total_chunks = 0
    with merged_path.open("w", encoding="utf-8") as out:
        for shard_run_id in shard_run_ids:
            chunk_file = PROCESSED_DIR / shard_run_id / "chunks" / "chunks.jsonl"
            if not chunk_file.exists():
                continue
            for line in chunk_file.open(encoding="utf-8"):
                line = line.strip()
                if line:
                    out.write(line + "\n")
                    total_chunks += 1

    print(f"Merged {total_chunks} chunks → {merged_path}")
    print(f"Done at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
