#!/usr/bin/env python3
"""
Re-parse vendor batch response to extract per-vendor signal scores.

The original run_phase2_classifiers.py discarded per-vendor signals,
saving only a scalar confidence. This script re-downloads the raw batch
response and patches vendor_signals into the existing testbed JSONL.

Usage:
    python3 scripts/repatch_vendor_signals.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RUNS_DIR = REPO_ROOT / "data" / "testbed_runs"

RUN_ID = "p2-vendor-gemini-3-flash-preview-full-v1"


def main() -> None:
    meta_path = RUNS_DIR / f"{RUN_ID}.batch_meta.json"
    testbed_path = RUNS_DIR / f"{RUN_ID}.jsonl"

    if not meta_path.exists():
        sys.exit(f"Missing batch meta: {meta_path}")
    if not testbed_path.exists():
        sys.exit(f"Missing testbed JSONL: {testbed_path}")

    meta = json.loads(meta_path.read_text())
    job_name = meta["job_name"]
    print(f"Batch job: {job_name}")

    # Download raw batch response
    sys.path.insert(0, str(REPO_ROOT / "pipeline"))
    from src.utils.batch_api import BatchClient

    batch = BatchClient(runs_dir=RUNS_DIR)
    job = batch.client.batches.get(name=job_name)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        sys.exit(f"Job state: {job.state.name} — cannot re-parse")

    output_lines: list[dict] = []
    dest = getattr(job, "dest", None)
    result_file_name = getattr(dest, "file_name", None) if dest else None

    if result_file_name:
        raw_bytes = batch.client.files.download(file=result_file_name)
        output_text = raw_bytes.decode("utf-8")
        output_lines = [json.loads(line) for line in output_text.splitlines() if line.strip()]
        print(f"Downloaded {len(output_lines)} raw responses")

    if not output_lines and dest and getattr(dest, "inlined_responses", None):
        for ir in dest.inlined_responses:
            entry: dict = {}
            key = getattr(ir, "key", None)
            if key is not None:
                entry["key"] = str(key)
            resp = getattr(ir, "response", None)
            if resp:
                entry["response"] = resp.model_dump() if hasattr(resp, "model_dump") else {}
            output_lines.append(entry)
        print(f"Extracted {len(output_lines)} inlined responses")

    if not output_lines:
        sys.exit("No responses found")

    # Build key -> vendor_signals map
    vendor_signals_by_key: dict[str, dict[str, int]] = {}
    for entry in output_lines:
        key = entry.get("key")
        if key is None:
            continue

        response_obj = entry.get("response", {})
        candidates = response_obj.get("candidates", [])
        if not candidates:
            continue

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            continue

        response_text = parts[0].get("text")
        if not response_text:
            continue

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            continue

        vendors = parsed.get("vendors", [])
        if not isinstance(vendors, list):
            continue

        signals: dict[str, int] = {}
        for v in vendors:
            if not isinstance(v, dict):
                continue
            tag = v.get("vendor", "")
            if isinstance(tag, dict):
                tag = tag.get("value", str(tag))
            tag = str(tag).strip().lower()
            signal = v.get("signal", 0)
            if isinstance(signal, (int, float)) and signal > 0:
                signals[tag] = max(signals.get(tag, 0), int(signal))

        vendor_signals_by_key[str(key)] = signals

    print(f"Extracted vendor signals for {len(vendor_signals_by_key)} responses")

    # Patch testbed JSONL
    records = []
    with testbed_path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    patched = 0
    for i, record in enumerate(records):
        signals = vendor_signals_by_key.get(str(i), {})
        if signals:
            record["vendor_signals"] = signals
            patched += 1
        else:
            record["vendor_signals"] = {}

    with testbed_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Patched {patched}/{len(records)} records in {testbed_path.name}")

    # Show sample
    for r in records[:3]:
        if r.get("vendor_signals"):
            print(f"  Sample: {r.get('company_name')} — {r['vendor_signals']}")


if __name__ == "__main__":
    main()
