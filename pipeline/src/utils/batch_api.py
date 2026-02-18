"""
Gemini Batch API helper for async classification at 50% cost.

Uses file-based batch jobs with a `key` field on each request for
reliable response-to-request matching (no LLM echo needed).

Usage in classifier_testbed.py:
    from src.utils.batch_api import BatchClient

    batch = BatchClient()
    jsonl_path = batch.prepare_requests(run_id="my-run", chunks=chunks)
    job_name = batch.submit(run_id="my-run", jsonl_path=jsonl_path)

    # Later (or in another cell):
    status = batch.check_status(job_name)
    if status["state"] == "JOB_STATE_SUCCEEDED":
        results = batch.get_results(job_name, chunks)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


class BatchClient:
    """Client for Gemini Batch API operations (file-based)."""

    def __init__(
        self,
        api_key: str | None = None,
        runs_dir: Path | None = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")

        self.client = genai.Client(api_key=self.api_key)
        self.runs_dir = runs_dir

        self._user_template = (
            "## EXCERPT\n"
            "\"\"\"\n"
            "{text}\n"
            "\"\"\"\n"
        )

    def _get_prompt_for_chunk(self, chunk: dict) -> tuple[str, str]:
        """Build system and user prompts for a chunk (mirrors LLMClassifierV2)."""
        from src.utils.prompt_loader import get_prompt_messages

        text = chunk["chunk_text"]
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        firm_name = chunk.get("company_name", "Unknown Company")
        report_year = chunk.get("report_year", "Unknown")
        sector = "Unknown"
        report_section = (
            chunk.get("report_sections", ["Unknown"])[0]
            if chunk.get("report_sections")
            else "Unknown"
        )

        return get_prompt_messages(
            "mention_type_v3",
            reasoning_policy="short",
            user_template=self._user_template,
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            report_section=report_section,
            text=text,
        )

    def _get_response_schema(self) -> dict:
        """Get cleaned response schema for Gemini."""
        from src.classifiers.schemas import MentionTypeResponseV2
        from src.classifiers.base_classifier import _clean_schema_for_gemini

        raw_schema = MentionTypeResponseV2.model_json_schema()
        return _clean_schema_for_gemini(raw_schema)

    def prepare_requests(
        self,
        run_id: str,
        chunks: list[dict],
        temperature: float = 0.0,
        thinking_budget: int | None = 0,
    ) -> Path:
        """
        Write batch requests to a JSONL file with key-based matching.

        Each line: {"key": "<index>", "request": {GenerateContentRequest}}

        Args:
            run_id: Used to name the JSONL file.
            chunks: List of chunk dicts.
            temperature: LLM temperature.
            thinking_budget: Token budget for thinking.
                Use `0` to explicitly disable thinking; use `None` to omit setting.

        Returns:
            Path to the written JSONL file.
        """
        if not self.runs_dir:
            raise RuntimeError("runs_dir must be set to write batch JSONL files")

        response_schema = self._get_response_schema()
        jsonl_path = self.runs_dir / f"{run_id}.batch_input.jsonl"

        with jsonl_path.open("w") as f:
            for i, chunk in enumerate(chunks):
                system_prompt, user_prompt = self._get_prompt_for_chunk(chunk)

                generation_config: dict[str, Any] = {
                    "temperature": temperature,
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                }
                if thinking_budget is not None:
                    generation_config["thinking_config"] = {
                        "thinking_budget": int(thinking_budget)
                    }

                line = {
                    "key": str(i),
                    "request": {
                        "contents": [{"parts": [{"text": user_prompt}], "role": "user"}],
                        "system_instruction": {"parts": [{"text": system_prompt}]},
                        "generation_config": generation_config,
                    },
                }
                f.write(json.dumps(line) + "\n")

        print(f"Wrote {len(chunks)} batch requests to {jsonl_path.name}")
        return jsonl_path

    def submit(
        self,
        run_id: str,
        jsonl_path: Path,
        model_name: str = "gemini-2.0-flash",
    ) -> str:
        """
        Upload JSONL file and submit a file-based batch job.

        Args:
            run_id: Identifier for this batch (used as display_name).
            jsonl_path: Path to JSONL file from prepare_requests().
            model_name: Model to use.

        Returns:
            Batch job name (use to check status and get results).
        """
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        # Upload input file via Files API
        uploaded = self.client.files.upload(
            file=str(jsonl_path),
            config=types.UploadFileConfig(
                display_name=run_id,
                mime_type="jsonl",
            ),
        )
        print(f"Uploaded: {uploaded.name}")

        # Create batch job referencing the uploaded file
        batch_job = self.client.batches.create(
            model=model_name,
            src=uploaded.name,
            config={"display_name": run_id},
        )

        job_name = batch_job.name
        num_requests = sum(1 for _ in jsonl_path.open())
        print(f"Submitted batch: {job_name}")
        print(f"  run_id: {run_id}")
        print(f"  model: {model_name}")
        print(f"  requests: {num_requests}")
        print(f"  state: {batch_job.state.name}")

        # Save metadata
        if self.runs_dir:
            meta_path = self.runs_dir / f"{run_id}.batch_meta.json"
            meta = {
                "job_name": job_name,
                "run_id": run_id,
                "model_name": model_name,
                "num_requests": num_requests,
                "uploaded_file": uploaded.name,
                "submitted_at": datetime.now().isoformat(),
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"  metadata: {meta_path.name}")

        return job_name

    def check_status(self, job_name: str) -> dict:
        """Check batch job status."""
        job = self.client.batches.get(name=job_name)

        status = {
            "job_name": job_name,
            "state": job.state.name,
            "display_name": getattr(job, "display_name", None),
        }

        if hasattr(job, "batch_stats") and job.batch_stats:
            stats = job.batch_stats
            status["total"] = getattr(stats, "total_count", None)
            status["succeeded"] = getattr(stats, "success_count", None)
            status["failed"] = getattr(stats, "failed_count", None)

        state = status["state"]
        if state == "JOB_STATE_SUCCEEDED":
            print(f"SUCCEEDED: {job_name}")
        elif state == "JOB_STATE_FAILED":
            print(f"FAILED: {job_name}")
        elif state == "JOB_STATE_CANCELLED":
            print(f"CANCELLED: {job_name}")
        elif state in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING"):
            label = state.replace("JOB_STATE_", "")
            print(f"{label}: {job_name}")
            if status.get("total"):
                done = status.get("succeeded", 0) or 0
                pct = done / status["total"] * 100
                print(f"  progress: {done}/{status['total']} ({pct:.0f}%)")
        else:
            print(f"? {state}: {job_name}")

        return status

    def list_jobs(self, limit: int = 10) -> list[dict]:
        """List recent batch jobs."""
        jobs = []
        for job in self.client.batches.list():
            jobs.append({
                "name": job.name,
                "display_name": getattr(job, "display_name", None),
                "state": job.state.name,
            })
            if len(jobs) >= limit:
                break

        print(f"Recent batch jobs ({len(jobs)}):")
        for j in jobs:
            print(f"  {j['state']:20s} {j['display_name'] or j['name']}")

        return jobs

    def get_results(
        self,
        job_name: str,
        chunks: list[dict],
    ) -> list[dict] | None:
        """
        Download results from a completed file-based batch and match by key.

        Args:
            job_name: The batch job name.
            chunks: Original chunks list (indexed 0..N-1, matching keys).

        Returns:
            List of result dicts in original chunk order, or None if not ready.
        """
        job = self.client.batches.get(name=job_name)

        if job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"Batch not ready: {job.state.name}")
            return None

        # Extract responses: try file download first, fall back to inlined
        output_lines: list[dict] = []
        dest = getattr(job, "dest", None)

        result_file_name = getattr(dest, "file_name", None) if dest else None
        if result_file_name:
            raw_bytes = self.client.files.download(file=result_file_name)
            output_text = raw_bytes.decode("utf-8")
            output_lines = [
                json.loads(line) for line in output_text.splitlines() if line.strip()
            ]
            print(f"Downloaded {len(output_lines)} responses from {result_file_name}")

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
            print("ERROR: No responses found (no file output, no inlined responses)")
            return None

        if len(output_lines) != len(chunks):
            print(f"WARNING: {len(output_lines)} responses != {len(chunks)} chunks")

        # Build lookup by key
        response_by_key: dict[str, dict] = {}
        missing_key_count = 0
        for entry in output_lines:
            key = entry.get("key")
            if key is not None:
                response_by_key[str(key)] = entry
            else:
                missing_key_count += 1

        # If keys are missing entirely, fall back to positional matching
        if not response_by_key and output_lines:
            print("WARNING: No response keys found; falling back to positional matching.")
            response_by_key = {str(i): entry for i, entry in enumerate(output_lines)}
        elif missing_key_count:
            print(f"WARNING: {missing_key_count} responses missing keys; unmatched entries may be skipped.")

        # Join in original order
        results = []
        matched = 0
        errors = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", chunk.get("annotation_id", f"unknown_{i}"))
            entry = response_by_key.get(str(i))

            if entry is None:
                error = f"Key '{i}': no response returned"
                errors.append(error)
                results.append(self._make_error_result(chunk, chunk_id, error))
                continue

            # Check for API-level error on this request
            if "error" in entry and entry["error"]:
                error = f"Key '{i}': API error: {entry['error']}"
                errors.append(error)
                results.append(self._make_error_result(chunk, chunk_id, error))
                continue

            try:
                response_obj = entry.get("response", {})
                # Extract text from response candidates
                response_text = None
                candidates = response_obj.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        response_text = parts[0].get("text")

                if not response_text:
                    error = f"Key '{i}': empty response text"
                    errors.append(error)
                    results.append(self._make_error_result(chunk, chunk_id, error))
                    continue

                parsed = json.loads(response_text)
                llm_types = [str(mt) for mt in parsed.get("mention_types", [])]

                confidence = 0.0
                if "confidence_scores" in parsed:
                    scores = parsed["confidence_scores"]
                    valid = [v for v in scores.values() if isinstance(v, (int, float))]
                    confidence = max(valid) if valid else 0.0

                results.append({
                    "chunk_id": chunk_id,
                    "company_name": chunk.get("company_name", "Unknown"),
                    "report_year": chunk.get("report_year", 0),
                    "human_mention_types": chunk.get("mention_types", []),
                    "llm_mention_types": llm_types,
                    "confidence": confidence,
                    "reasoning": parsed.get("reasoning", ""),
                    "chunk_text": chunk.get("chunk_text", ""),
                })
                matched += 1

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                error = f"Key '{i}': parse error: {e}"
                errors.append(error)
                results.append(self._make_error_result(chunk, chunk_id, error))

        print(f"Matched: {matched}/{len(chunks)} | Errors: {len(errors)}")
        for err in errors:
            print(f"  - {err}")

        return results

    def _make_error_result(self, chunk: dict, chunk_id: str, error: str) -> dict:
        """Create an error result entry."""
        return {
            "chunk_id": chunk_id,
            "company_name": chunk.get("company_name", "Unknown"),
            "report_year": chunk.get("report_year", 0),
            "human_mention_types": chunk.get("mention_types", []),
            "llm_mention_types": [],
            "confidence": 0.0,
            "reasoning": error,
            "chunk_text": chunk.get("chunk_text", ""),
            "error": error,
        }

    def poll_until_complete(
        self,
        jobs: dict[str, str],
        interval: int = 30,
        max_time: int = 86400,
    ) -> dict[str, dict]:
        """Poll multiple batch jobs until all reach a terminal state.

        Args:
            jobs: Mapping of label (e.g. classifier name) to job_name.
            interval: Seconds between polling rounds.
            max_time: Maximum total wait time in seconds (default 24h).

        Returns:
            Dict of {label: final_status_dict} for every job.
        """
        import time as _time

        start = _time.time()
        terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
        completed: dict[str, dict] = {}

        while True:
            pending = []
            for label, job_name in jobs.items():
                if label in completed:
                    continue
                try:
                    status = self.check_status(job_name)
                except Exception as poll_err:
                    print(f"WARNING: poll error for {label}: {poll_err!r} — will retry next cycle")
                    pending.append(label)
                    continue
                if status["state"] in terminal_states:
                    completed[label] = status
                else:
                    pending.append(label)

            if not pending:
                print(f"\nAll {len(jobs)} jobs complete.")
                break

            elapsed = _time.time() - start
            if elapsed > max_time:
                print(f"\nTimeout after {elapsed:.0f}s. Still pending: {pending}")
                for label in pending:
                    completed[label] = {"state": "TIMEOUT", "job_name": jobs[label]}
                break

            print(
                f"\nWaiting {interval}s... "
                f"({len(pending)} pending, {len(completed)} done, {elapsed:.0f}s elapsed)"
            )
            _time.sleep(interval)

        return completed

    def cancel(self, job_name: str) -> None:
        """Cancel a running batch job."""
        self.client.batches.cancel(name=job_name)
        print(f"Cancelled: {job_name}")
