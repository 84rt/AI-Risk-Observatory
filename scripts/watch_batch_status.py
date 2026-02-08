#!/usr/bin/env python3
"""Watch a Gemini Batch job until it's ready for download."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


ANSI_RESET = "\x1b[0m"
ANSI_GREEN = "\x1b[32m"
ANSI_YELLOW = "\x1b[33m"
ANSI_RED = "\x1b[31m"


def colorize(text: str, color: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"{color}{text}{ANSI_RESET}"


def format_timedelta(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def load_env_local(repo_root: Path) -> None:
    env_path = repo_root / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll a Gemini Batch job and print status every interval until ready."
    )
    parser.add_argument("--job-name", type=str, help="Batch job name (from submit())")
    parser.add_argument("--run-id", type=str, help="Run id to resolve job name from batch meta")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("data/testbed_runs"),
        help="Directory containing <run_id>.batch_meta.json",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=120,
        help="Polling interval in seconds",
    )
    return parser.parse_args()


def resolve_job_name(args: argparse.Namespace) -> str:
    if args.job_name:
        return args.job_name
    if not args.run_id:
        raise SystemExit("Provide --job-name or --run-id.")
    meta_path = args.runs_dir / f"{args.run_id}.batch_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Batch meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    job_name = meta.get("job_name")
    if not job_name:
        raise SystemExit(f"No job_name in {meta_path}")
    return job_name


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    pipeline_utils = repo_root / "pipeline" / "src" / "utils" / "batch_api.py"
    if not pipeline_utils.exists():
        raise SystemExit(f"Batch API not found: {pipeline_utils}")

    load_env_local(repo_root)

    # Load BatchClient directly to avoid importing pipeline config.
    import importlib.util

    spec = importlib.util.spec_from_file_location("batch_api", pipeline_utils)
    if not spec or not spec.loader:
        raise SystemExit("Failed to load batch_api module.")
    batch_api = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(batch_api)
    except ModuleNotFoundError as exc:
        if exc.name in {"google", "google.genai"}:
            raise SystemExit(
                "Missing dependency: google-genai. Install with "
                "`python3 -m pip install google-genai`, or run with a Python "
                "environment that already has it."
            ) from exc
        raise
    BatchClient = batch_api.BatchClient

    job_name = resolve_job_name(args)
    client = BatchClient(runs_dir=args.runs_dir)
    started_at = datetime.now()
    print(f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    while True:
        status = client.check_status(job_name)
        state = status.get("state")
        timestamp = datetime.now().strftime("%H:%M:%S")

        if state == "JOB_STATE_SUCCEEDED":
            msg = colorize("✅ ready for download", ANSI_GREEN)
            print(f"[{timestamp}] {msg}", flush=True)
            elapsed = int((datetime.now() - started_at).total_seconds())
            print(f"Elapsed: {format_timedelta(elapsed)}", flush=True)
            break
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            msg = colorize(f"❌ {state.lower()}", ANSI_RED)
            print(f"[{timestamp}] {msg}", flush=True)
            elapsed = int((datetime.now() - started_at).total_seconds())
            print(f"Elapsed: {format_timedelta(elapsed)}", flush=True)
            break
        if state in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING"):
            msg = colorize("⏳ pending", ANSI_YELLOW)
            print(f"[{timestamp}] {msg}", flush=True)
        else:
            msg = colorize(f"❓ {state}", ANSI_YELLOW)
            print(f"[{timestamp}] {msg}", flush=True)

        time.sleep(max(5, args.interval_seconds))


if __name__ == "__main__":
    main()
