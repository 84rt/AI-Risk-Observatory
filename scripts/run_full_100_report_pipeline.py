#!/usr/bin/env python3
"""
Orchestrate end-to-end LLM annotation for the 100-report expansion workflow.

Workflow:
1) Preflight checks (queue coverage + chunk inventory)
2) Phase 1 mention-type classification via Gemini Batch API
3) Phase 2 classifiers (risk, adoption_type, vendor) via existing runner
4) Merge phase-2 outputs into one final annotations JSONL

This script is transparency-first: each step prints inputs, outputs, and run IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
RUNS_DIR = REPO_ROOT / "data" / "testbed_runs"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

BatchClient = None
MentionTypeClassifier = None
_clean_schema_for_gemini = None


DEFAULT_QUEUE = REPO_ROOT / "data" / "processing_queue.json"
DEFAULT_COMPANIES = REPO_ROOT / "data" / "reference" / "companies_metadata_v2.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "golden_set" / "phase2_annotated"


@dataclass
class PreflightSummary:
    expected_reports: int
    queued_reports: int
    missing_reports: list[dict[str, Any]]
    chunk_count: int
    chunk_reports: int
    chunk_companies: int
    chunk_years: list[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full 100-report LLM workflow (phase1 mention + phase2 + merge)."
    )
    p.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Path to chunk JSONL input.",
    )
    p.add_argument(
        "--processed-run-id",
        type=str,
        default=None,
        help="If --chunks is omitted, resolve from data/processed/<run_id>/...",
    )
    p.add_argument(
        "--chunks-subpath",
        type=str,
        default="chunks-keyword-casefix-v2/chunks.jsonl",
        help="Relative path under processed run directory (used with --processed-run-id).",
    )
    p.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model name.",
    )
    p.add_argument(
        "--run-suffix",
        required=True,
        help="Shared run suffix used for phase1 and phase2 outputs (e.g. full-100-v1).",
    )
    p.add_argument(
        "--phase2-classifiers",
        default="risk,adoption_type,vendor",
        help="Comma-separated phase2 classifiers.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for all batch requests.",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help=(
            "Thinking budget for Gemini. Set 0 to explicitly disable thinking "
            "and preserve output tokens for JSON."
        ),
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Batch polling interval (seconds).",
    )
    p.add_argument(
        "--max-poll-time",
        type=int,
        default=86400,
        help="Max polling time in seconds before timeout (default: 86400 = 24h).",
    )
    p.add_argument(
        "--processing-queue",
        type=Path,
        default=DEFAULT_QUEUE,
        help="Queue JSON to validate report coverage.",
    )
    p.add_argument(
        "--companies-manifest",
        type=Path,
        default=DEFAULT_COMPANIES,
        help="CSV manifest with company_name + lei.",
    )
    p.add_argument(
        "--expected-years",
        default="2023,2024",
        help="Comma-separated report years expected per company.",
    )
    p.add_argument(
        "--allow-report-mismatch",
        action="store_true",
        help="Do not fail preflight when queue coverage is below expected.",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight checks.",
    )
    p.add_argument(
        "--resume-phase1",
        action="store_true",
        help="Resume phase1 from existing <run_id>.batch_meta.json instead of submitting a new job.",
    )
    p.add_argument(
        "--phase1-output-jsonl",
        type=Path,
        default=None,
        help=(
            "Local batch output JSONL for phase1 (offline parse mode). "
            "If set, skips submit/poll/download and parses this file directly."
        ),
    )
    p.add_argument(
        "--resume-phase2",
        action="store_true",
        help="Run phase2 runner with --resume.",
    )
    p.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Run only preflight + phase1 and stop before phase2 merge.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for phase1 annotations, merged output, and manifest.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare/check only; do not submit or poll batch jobs.",
    )
    return p.parse_args()


def load_env() -> None:
    env_path = REPO_ROOT / ".env.local"
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


def init_pipeline_imports() -> None:
    global BatchClient, MentionTypeClassifier, _clean_schema_for_gemini
    if BatchClient is not None and MentionTypeClassifier is not None:
        return

    from src.classifiers.base_classifier import _clean_schema_for_gemini as _schema_cleaner
    from src.classifiers.mention_type_classifier import MentionTypeClassifier as _mention_cls
    from src.utils.batch_api import BatchClient as _batch_client

    BatchClient = _batch_client
    MentionTypeClassifier = _mention_cls
    _clean_schema_for_gemini = _schema_cleaner


def run_cmd(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def resolve_chunks_path(args: argparse.Namespace) -> Path:
    if args.chunks:
        return args.chunks
    if not args.processed_run_id:
        raise SystemExit("Provide either --chunks or --processed-run-id.")

    base = REPO_ROOT / "data" / "processed" / args.processed_run_id
    direct = base / args.chunks_subpath
    if direct.exists():
        return direct

    fallbacks = [
        base / "chunks-keyword-casefix-v2" / "chunks.jsonl",
        base / "chunks" / "chunks.jsonl",
    ]
    for path in fallbacks:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not resolve chunks path under {base}. Tried: {direct}, {fallbacks}"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_years(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("No valid years parsed from --expected-years")
    return sorted(set(out))


def preflight(
    chunks: list[dict[str, Any]],
    queue_path: Path,
    companies_path: Path,
    expected_years: list[int],
) -> PreflightSummary:
    expected_pairs: set[tuple[str, int, str]] = set()
    if companies_path.exists():
        with companies_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                lei = (row.get("lei") or "").strip()
                name = (row.get("company_name") or "").strip()
                if not lei:
                    continue
                for year in expected_years:
                    expected_pairs.add((lei, year, name))

    queue_pairs: set[tuple[str, int]] = set()
    if queue_path.exists():
        payload = json.loads(queue_path.read_text())
        for row in payload:
            lei = str(row.get("lei") or "").strip()
            year = row.get("year")
            if not lei or not isinstance(year, int):
                continue
            queue_pairs.add((lei, year))

    missing: list[dict[str, Any]] = []
    for lei, year, name in sorted(expected_pairs, key=lambda x: (x[2], x[1])):
        if (lei, year) not in queue_pairs:
            missing.append({"company_name": name, "lei": lei, "year": year})

    chunk_docs = {str(c.get("document_id")) for c in chunks if c.get("document_id")}
    chunk_companies = {
        str(c.get("company_name"))
        for c in chunks
        if c.get("company_name")
    }
    chunk_years = sorted(
        {
            int(c.get("report_year"))
            for c in chunks
            if isinstance(c.get("report_year"), int)
        }
    )

    return PreflightSummary(
        expected_reports=len(expected_pairs),
        queued_reports=len(queue_pairs),
        missing_reports=missing,
        chunk_count=len(chunks),
        chunk_reports=len(chunk_docs),
        chunk_companies=len(chunk_companies),
        chunk_years=chunk_years,
    )


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "firm_id": chunk.get("company_id", "Unknown"),
        "firm_name": chunk.get("company_name", "Unknown"),
        "report_year": chunk.get("report_year", 0),
        "sector": chunk.get("sector", "Unknown") or "Unknown",
        "report_section": (
            chunk.get("report_sections", ["Unknown"])[0]
            if chunk.get("report_sections")
            else "Unknown"
        ),
    }


def normalize_label_token(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    token = str(value).strip()
    if token.startswith("MentionType."):
        return token.split(".", 1)[1]
    return token


def try_parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    attempts: list[str] = [text]
    stripped = text.strip()
    if stripped.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        attempts.append(cleaned.strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        attempts.append(stripped[start : end + 1])
    for candidate in list(attempts):
        attempts.append(re.sub(r",\s*([}\]])", r"\1", candidate))
    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def try_parse_json_with_error(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Like try_parse_json, but returns a best parse error detail when it fails."""
    if not text:
        return None, "empty response text"

    attempts: list[str] = [text]
    stripped = text.strip()
    if stripped.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        attempts.append(cleaned.strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        attempts.append(stripped[start : end + 1])
    for candidate in list(attempts):
        attempts.append(re.sub(r",\s*([}\]])", r"\1", candidate))

    first_error: str | None = None
    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError as exc:
            if first_error is None:
                first_error = str(exc)
            continue

    return None, first_error or "unable to parse model JSON response"


def salvage_phase1_payload(text: str) -> dict[str, Any] | None:
    """Best-effort extraction for malformed phase1 mention-type payloads."""
    if not text:
        return None

    labels = {"adoption", "risk", "vendor", "general_ambiguous", "none"}
    payload: dict[str, Any] = {}

    mention_types: list[str] = []
    m_types = re.search(r'"mention_types"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    if m_types:
        tokens = re.findall(r'"?([a-z_]+)"?', m_types.group(1))
        mention_types = [t for t in tokens if t in labels]

    confidence_scores: dict[str, float] = {}
    m_conf = re.search(r'"confidence_scores"\s*:\s*\{(.*?)\}', text, flags=re.DOTALL)
    if m_conf:
        for k, v in re.findall(
            r'"?([a-z_]+)"?\s*:\s*(-?\d+(?:\.\d+)?)',
            m_conf.group(1),
            flags=re.DOTALL,
        ):
            if k in labels:
                try:
                    confidence_scores[k] = float(v)
                except ValueError:
                    continue

    # Fallback for truncated mention_types list without closing bracket.
    if not mention_types:
        m_types_prefix = re.search(r'"mention_types"\s*:\s*\[(.*)', text, flags=re.DOTALL)
        if m_types_prefix:
            tokens = re.findall(r'"(adoption|risk|vendor|general_ambiguous|none)"', m_types_prefix.group(1))
            mention_types = list(tokens)

    if not mention_types and confidence_scores:
        mention_types = list(confidence_scores.keys())

    m_reason = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, flags=re.DOTALL)
    reasoning = m_reason.group(1) if m_reason else ""

    mention_types = sorted(set(mention_types))
    if "none" in mention_types and len(mention_types) > 1:
        mention_types = [t for t in mention_types if t != "none"]

    if not mention_types:
        return None

    payload["mention_types"] = mention_types
    if confidence_scores:
        payload["confidence_scores"] = confidence_scores
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def prepare_phase1_batch_jsonl(
    run_id: str,
    chunks: list[dict[str, Any]],
    temperature: float,
    thinking_budget: int | None = 0,
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{run_id}.batch_input.jsonl"
    temp_classifier = MentionTypeClassifier(run_id="batch-prep", model_name="unused")
    response_schema = _clean_schema_for_gemini(
        MentionTypeClassifier.RESPONSE_MODEL.model_json_schema()
    )

    with out_path.open("w") as f:
        for i, chunk in enumerate(chunks):
            system_prompt, user_prompt = temp_classifier.get_prompt_messages(
                chunk.get("chunk_text", ""),
                build_metadata(chunk),
            )
            line = {
                "key": str(i),
                "request": {
                    "contents": [{"parts": [{"text": user_prompt}], "role": "user"}],
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "generation_config": {
                        "temperature": temperature,
                        "max_output_tokens": 2048,
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                    },
                },
            }
            if thinking_budget is not None:
                line["request"]["generation_config"]["thinking_config"] = {
                    "thinking_budget": int(thinking_budget)
                }
            f.write(json.dumps(line) + "\n")
    return out_path


def download_output_lines(batch: BatchClient, job_name: str) -> list[dict[str, Any]]:
    job = batch.client.batches.get(name=job_name)
    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Phase1 batch job is not succeeded: {job.state.name}")

    lines: list[dict[str, Any]] = []
    dest = getattr(job, "dest", None)
    result_file_name = getattr(dest, "file_name", None) if dest else None

    if result_file_name:
        raw_bytes = batch.client.files.download(file=result_file_name)
        output_text = raw_bytes.decode("utf-8")
        lines = [json.loads(line) for line in output_text.splitlines() if line.strip()]
        print(f"Downloaded {len(lines)} phase1 responses from {result_file_name}")

    if not lines and dest and getattr(dest, "inlined_responses", None):
        for ir in dest.inlined_responses:
            entry: dict[str, Any] = {}
            key = getattr(ir, "key", None)
            if key is not None:
                entry["key"] = str(key)
            resp = getattr(ir, "response", None)
            if resp:
                entry["response"] = resp.model_dump() if hasattr(resp, "model_dump") else {}
            lines.append(entry)
        print(f"Extracted {len(lines)} inlined phase1 responses")

    if not lines:
        raise RuntimeError("No phase1 responses found in batch output.")
    return lines


def load_output_lines_cache(path: Path) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
    return lines


def save_output_lines_cache(path: Path, lines: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in lines:
            f.write(json.dumps(row) + "\n")
    print(f"Cached phase1 batch output: {path}")


def parse_phase1_results(
    chunks: list[dict[str, Any]],
    output_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    response_by_key: dict[str, dict[str, Any]] = {}
    for entry in output_lines:
        key = entry.get("key")
        if key is not None:
            response_by_key[str(key)] = entry
    if not response_by_key:
        response_by_key = {str(i): e for i, e in enumerate(output_lines)}

    results: list[dict[str, Any]] = []
    matched = 0
    salvaged = 0
    for i, chunk in enumerate(chunks):
        chunk_id = str(chunk.get("chunk_id") or f"unknown_{i}")
        base = {
            "chunk_id": chunk_id,
            "company_name": chunk.get("company_name", "Unknown"),
            "report_year": chunk.get("report_year", 0),
            "llm_mention_types": [],
            "confidence": 0.0,
            "mention_confidences": {},
            "reasoning": "",
            "chunk_text": chunk.get("chunk_text", ""),
            "error": None,
            "response_preview": None,
        }

        entry = response_by_key.get(str(i))
        if entry is None:
            base["error"] = f"Key '{i}': no response returned"
            results.append(base)
            continue
        if entry.get("error"):
            base["error"] = f"Key '{i}': API error: {entry.get('error')}"
            results.append(base)
            continue

        try:
            response_obj = entry.get("response", {})
            candidates = response_obj.get("candidates", [])
            response_text = None
            for cand in candidates:
                parts = cand.get("content", {}).get("parts", [])
                text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
                if text_parts:
                    response_text = "\n".join(text_parts).strip()
                    break
            parsed, parse_err = try_parse_json_with_error(response_text or "")
            if parsed is None:
                parsed = salvage_phase1_payload(response_text or "")
                if parsed is not None:
                    salvaged += 1
            if parsed is None:
                raise ValueError(parse_err or "unable to parse model JSON response")

            raw_types = parsed.get("mention_types", [])
            if isinstance(raw_types, str):
                raw_types = [raw_types]
            mention_types = [
                normalize_label_token(t)
                for t in raw_types
                if normalize_label_token(t)
            ]
            mention_types = sorted(set(mention_types))
            if "none" in mention_types and len(mention_types) > 1:
                mention_types = [t for t in mention_types if t != "none"]
            if not mention_types:
                mention_types = ["none"]

            conf_raw = parsed.get("confidence_scores", {})
            conf_map: dict[str, float] = {}
            if isinstance(conf_raw, dict):
                for k, v in conf_raw.items():
                    if isinstance(v, (int, float)):
                        conf_map[normalize_label_token(k)] = float(v)
            confidence = max(conf_map.values()) if conf_map else 0.0

            base["llm_mention_types"] = mention_types
            base["mention_confidences"] = conf_map
            base["confidence"] = confidence
            base["reasoning"] = parsed.get("reasoning", "") or ""
            matched += 1
        except (ValueError, KeyError, IndexError) as exc:
            base["error"] = f"Key '{i}': parse error: {exc}"
            if response_text:
                preview = response_text.strip().replace("\n", "\\n")
                base["response_preview"] = preview[:800]

        results.append(base)

    print(f"Phase1 parsed successfully: {matched}/{len(chunks)}")
    if salvaged:
        print(f"Phase1 salvaged malformed payloads: {salvaged}")
    return results


def save_phase1_run(
    run_id: str,
    results: list[dict[str, Any]],
    model: str,
    temperature: float,
    chunk_source: Path,
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{run_id}.jsonl"
    with run_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "classifier_name": "mention_type",
            "model_name": model,
            "temperature": temperature,
            "batch_mode": True,
            "num_chunks": len(results),
            "schema_version": MentionTypeClassifier.SCHEMA_VERSION,
            "prompt_key": MentionTypeClassifier.PROMPT_KEY,
            "response_schema": MentionTypeClassifier.RESPONSE_MODEL.__name__,
            "chunk_source": str(chunk_source),
        },
        "num_chunks": len(results),
    }
    run_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote phase1 run: {run_path}")
    return run_path


def write_phase1_parse_debug(run_id: str, results: list[dict[str, Any]]) -> Path | None:
    debug_rows = [
        {
            "chunk_id": r.get("chunk_id"),
            "company_name": r.get("company_name"),
            "report_year": r.get("report_year"),
            "error": r.get("error"),
            "response_preview": r.get("response_preview"),
        }
        for r in results
        if r.get("error")
    ]
    if not debug_rows:
        return None

    path = RUNS_DIR / f"{run_id}.parse_failures.debug.jsonl"
    with path.open("w") as f:
        for row in debug_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote phase1 parse debug: {path}")
    return path


def write_phase1_annotations(
    chunks: list[dict[str, Any]],
    phase1_results: list[dict[str, Any]],
    output_path: Path,
    run_id: str,
    model: str,
) -> Path:
    by_chunk = {str(r.get("chunk_id")): r for r in phase1_results}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    with output_path.open("w") as f:
        for idx, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("chunk_id") or f"unknown_{idx}")
            p1 = by_chunk.get(chunk_id, {})
            mention_types = p1.get("llm_mention_types") or ["none"]

            record = {
                "annotation_id": f"llm-{run_id}-{chunk_id}",
                "run_id": run_id,
                "chunk_id": chunk_id,
                "document_id": chunk.get("document_id"),
                "company_id": chunk.get("company_id"),
                "company_name": chunk.get("company_name"),
                "report_year": chunk.get("report_year"),
                "report_sections": chunk.get("report_sections"),
                "chunk_text": chunk.get("chunk_text"),
                "matched_keywords": chunk.get("matched_keywords"),
                "created_at": now,
                "source": "llm",
                "classifier_id": "llm_testbed_mention_type",
                "classifier_version": "v2",
                "model_name": model,
                "schema_version": MentionTypeClassifier.SCHEMA_VERSION,
                "prompt_key": MentionTypeClassifier.PROMPT_KEY,
                "mention_types": mention_types,
                "adoption_types": [],
                "adoption_confidence": {},
                "risk_taxonomy": [],
                "risk_confidence": {},
                "risk_signals": [],
                "risk_substantiveness": None,
                "vendor_tags": [],
                "vendor_other": None,
                "vendor_confidence": {},
                "llm_details": {
                    "model": model,
                    "mention_confidences": p1.get("mention_confidences", {}),
                    "reasoning": p1.get("reasoning", ""),
                    "phase1_confidence": p1.get("confidence", 0.0),
                    "phase1_error": p1.get("error"),
                    "phase1_response_preview": p1.get("response_preview"),
                },
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote phase1 annotations: {output_path}")
    return output_path


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote manifest: {path}")


def main() -> None:
    args = parse_args()
    load_env()
    init_pipeline_imports()

    chunks_path = resolve_chunks_path(args)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    chunks = load_jsonl(chunks_path)
    if not chunks:
        raise SystemExit(f"No chunks found in {chunks_path}")

    expected_years = parse_years(args.expected_years)
    print("=" * 60)
    print("FULL PIPELINE ORCHESTRATOR")
    print("=" * 60)
    print(f"Chunks:        {chunks_path}")
    print(f"Chunks count:  {len(chunks)}")
    print(f"Model:         {args.model}")
    print(f"Run suffix:    {args.run_suffix}")
    print(f"Dry run:       {args.dry_run}")
    print()

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "chunks_path": str(chunks_path),
        "steps": {},
    }

    # ------------------------------------------------------------------
    # Step 1: Preflight
    # ------------------------------------------------------------------
    if not args.skip_preflight:
        summary = preflight(
            chunks=chunks,
            queue_path=args.processing_queue,
            companies_path=args.companies_manifest,
            expected_years=expected_years,
        )

        print("Preflight")
        print(f"  Expected reports: {summary.expected_reports}")
        print(f"  Queued reports:   {summary.queued_reports}")
        print(f"  Missing reports:  {len(summary.missing_reports)}")
        print(f"  Chunk docs:       {summary.chunk_reports}")
        print(f"  Chunk companies:  {summary.chunk_companies}")
        print(f"  Chunk years:      {summary.chunk_years}")
        if summary.missing_reports:
            print("  Missing queue pairs:")
            for item in summary.missing_reports[:20]:
                print(
                    f"    - {item['company_name']} | {item['year']} | {item['lei']}"
                )
            if len(summary.missing_reports) > 20:
                print(f"    ... and {len(summary.missing_reports) - 20} more")

        manifest["steps"]["preflight"] = {
            "expected_reports": summary.expected_reports,
            "queued_reports": summary.queued_reports,
            "missing_reports": summary.missing_reports,
            "chunk_count": summary.chunk_count,
            "chunk_reports": summary.chunk_reports,
            "chunk_companies": summary.chunk_companies,
            "chunk_years": summary.chunk_years,
        }

        if summary.missing_reports and not args.allow_report_mismatch:
            raise SystemExit(
                "Preflight failed: queue coverage below expected. "
                "Fix processing queue or rerun with --allow-report-mismatch."
            )

        print()

    # ------------------------------------------------------------------
    # Step 2: Phase1 mention-type batch
    # ------------------------------------------------------------------
    model_slug = slugify(args.model)
    p1_run_id = f"p1-mention_type-{model_slug}-{args.run_suffix}"
    p1_jsonl = prepare_phase1_batch_jsonl(
        run_id=p1_run_id,
        chunks=chunks,
        temperature=args.temperature,
        thinking_budget=args.thinking_budget,
    )
    print(f"Phase1 batch input: {p1_jsonl}")
    manifest["steps"]["phase1_prepare"] = {
        "run_id": p1_run_id,
        "batch_input_jsonl": str(p1_jsonl),
        "num_requests": len(chunks),
    }

    if args.dry_run:
        print("Dry run: stopping before batch submission.")
        manifest_path = args.output_dir / f"{p1_run_id}.manifest.json"
        save_manifest(manifest_path, manifest)
        return

    batch = BatchClient(runs_dir=RUNS_DIR)

    if args.phase1_output_jsonl:
        if not args.phase1_output_jsonl.exists():
            raise FileNotFoundError(f"--phase1-output-jsonl not found: {args.phase1_output_jsonl}")
        p1_job_name = "local-output-jsonl"
        print(f"Phase1 local output: {args.phase1_output_jsonl}")
    elif args.resume_phase1:
        meta_path = RUNS_DIR / f"{p1_run_id}.batch_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"--resume-phase1 set but batch meta not found: {meta_path}"
            )
        meta = json.loads(meta_path.read_text())
        p1_job_name = meta["job_name"]
        print(f"Phase1 resume: {p1_job_name}")
    else:
        p1_job_name = batch.submit(
            run_id=p1_run_id,
            jsonl_path=p1_jsonl,
            model_name=args.model,
        )
        print(f"Phase1 submitted: {p1_job_name}")

    output_cache_path = RUNS_DIR / f"{p1_run_id}.batch_output.jsonl"
    p1_state = None
    output_lines: list[dict[str, Any]] | None = None

    if args.phase1_output_jsonl:
        output_lines = load_output_lines_cache(args.phase1_output_jsonl)
        p1_state = "JOB_STATE_SUCCEEDED_LOCAL"
        save_output_lines_cache(output_cache_path, output_lines)

    # In resume mode, batch jobs may age out from API. Try direct fetch first.
    if output_lines is None and args.resume_phase1:
        try:
            output_lines = download_output_lines(batch=batch, job_name=p1_job_name)
            p1_state = "JOB_STATE_SUCCEEDED"
            save_output_lines_cache(output_cache_path, output_lines)
        except Exception as exc:
            err = str(exc)
            if "NOT_FOUND" in err and output_cache_path.exists():
                print(
                    "Phase1 batch job not found in API; using cached batch output "
                    f"from {output_cache_path}."
                )
                output_lines = load_output_lines_cache(output_cache_path)
                p1_state = "JOB_STATE_SUCCEEDED_CACHED"
            elif "NOT_FOUND" in err:
                raise SystemExit(
                    "Phase1 resume failed: batch job is not available in API and "
                    "no local batch output cache exists. Submit a retry run."
                ) from exc
            else:
                print(
                    "Phase1 direct download failed in resume mode; polling job state "
                    "and retrying download."
                )

    if output_lines is None:
        p1_status = batch.poll_until_complete(
            jobs={"mention_type": p1_job_name},
            interval=args.poll_interval,
            max_time=args.max_poll_time,
        )
        p1_state = p1_status.get("mention_type", {}).get("state")
        if p1_state != "JOB_STATE_SUCCEEDED":
            raise SystemExit(f"Phase1 did not succeed (state={p1_state})")

        output_lines = download_output_lines(batch=batch, job_name=p1_job_name)
        save_output_lines_cache(output_cache_path, output_lines)

    phase1_results = parse_phase1_results(chunks=chunks, output_lines=output_lines)
    phase1_run_path = save_phase1_run(
        run_id=p1_run_id,
        results=phase1_results,
        model=args.model,
        temperature=args.temperature,
        chunk_source=chunks_path,
    )
    phase1_parse_debug_path = write_phase1_parse_debug(
        run_id=p1_run_id,
        results=phase1_results,
    )

    phase1_annotations_path = args.output_dir / f"{p1_run_id}.phase1_annotations.jsonl"
    write_phase1_annotations(
        chunks=chunks,
        phase1_results=phase1_results,
        output_path=phase1_annotations_path,
        run_id=p1_run_id,
        model=args.model,
    )

    manifest["steps"]["phase1"] = {
        "run_id": p1_run_id,
        "job_name": p1_job_name,
        "state": p1_state,
        "phase1_run_path": str(phase1_run_path),
        "phase1_annotations_path": str(phase1_annotations_path),
        "phase1_parse_debug_path": str(phase1_parse_debug_path) if phase1_parse_debug_path else None,
    }

    if args.skip_phase2:
        manifest_path = args.output_dir / f"{p1_run_id}.manifest.json"
        save_manifest(manifest_path, manifest)
        print("Completed phase1 only (--skip-phase2).")
        return

    # ------------------------------------------------------------------
    # Step 3: Phase2 via existing production runner
    # ------------------------------------------------------------------
    phase2_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_phase2_classifiers.py"),
        "--classifiers",
        args.phase2_classifiers,
        "--model",
        args.model,
        "--run-suffix",
        args.run_suffix,
        "--golden-set",
        str(phase1_annotations_path),
        "--poll-interval",
        str(args.poll_interval),
        "--max-poll-time",
        str(args.max_poll_time),
        "--temperature",
        str(args.temperature),
        "--skip-export",
    ]
    if args.resume_phase2:
        phase2_cmd.append("--resume")
    run_cmd(phase2_cmd)

    # ------------------------------------------------------------------
    # Step 4: Merge phase2 outputs
    # ------------------------------------------------------------------
    final_output = (
        args.output_dir
        / f"p2-{model_slug}-{args.run_suffix}.annotations.jsonl"
    )
    merge_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "merge_phase2_into_golden_set.py"),
        "--golden-set",
        str(phase1_annotations_path),
        "--run-suffix",
        args.run_suffix,
        "--model",
        args.model,
        "--output",
        str(final_output),
    ]
    run_cmd(merge_cmd)

    manifest["steps"]["phase2"] = {
        "classifiers": args.phase2_classifiers,
        "run_suffix": args.run_suffix,
        "model": args.model,
    }
    manifest["steps"]["merge"] = {
        "output": str(final_output),
    }

    manifest_path = args.output_dir / f"full-{model_slug}-{args.run_suffix}.manifest.json"
    save_manifest(manifest_path, manifest)
    print(f"Final merged annotations: {final_output}")


if __name__ == "__main__":
    main()
