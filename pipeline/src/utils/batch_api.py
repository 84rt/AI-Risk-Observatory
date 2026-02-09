"""
Gemini Batch API helper for async classification at 50% cost.

Usage in classifier_testbed.py:
    from src.utils.batch_api import BatchClient

    batch = BatchClient()
    requests = batch.prepare_requests(chunks, model_name="gemini-2.0-flash")
    job_name = batch.submit(run_id="my-run", requests=requests)

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
    """Client for Gemini Batch API operations."""

    def __init__(
        self,
        api_key: str | None = None,
        runs_dir: Path | None = None,
    ):
        """
        Initialize the batch client.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            runs_dir: Directory to save batch metadata and results
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")

        self.client = genai.Client(api_key=self.api_key)
        self.runs_dir = runs_dir

        self._user_template = (
            "## CHUNK_INDEX\n"
            "{chunk_index}\n\n"
            "## NOTE\n"
            "Echo this integer as \"chunk_id\" in the JSON output.\n\n"
            "## EXCERPT\n"
            "\"\"\"\n"
            "{text}\n"
            "\"\"\"\n"
        )

    def _get_prompt_for_chunk(self, chunk: dict, chunk_index: int) -> tuple[str, str]:
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
            chunk_index=chunk_index,
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
        chunks: list[dict],
        temperature: float = 0.0,
        thinking_budget: int = 0,
    ) -> list[dict]:
        """
        Prepare batch requests from chunks.

        Args:
            chunks: List of chunk dicts with chunk_text, company_name, etc.
            temperature: LLM temperature (0.0 for deterministic)
            thinking_budget: Token budget for thinking (0 = disabled)

        Returns:
            List of request dicts for submit()
        """
        response_schema = self._get_response_schema()
        requests = []

        for i, chunk in enumerate(chunks):
            system_prompt, user_prompt = self._get_prompt_for_chunk(chunk, chunk_index=i)

            config: dict[str, Any] = {
                "system_instruction": system_prompt,
                "temperature": temperature,
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }

            if thinking_budget > 0:
                config["thinking_config"] = {"thinking_budget": thinking_budget}

            requests.append({
                "contents": [{"parts": [{"text": user_prompt}], "role": "user"}],
                "config": config,
            })

        print(f"Prepared {len(requests)} batch requests.")
        return requests

    def submit(
        self,
        run_id: str,
        requests: list[dict],
        model_name: str = "gemini-2.0-flash",
    ) -> str:
        """
        Submit a batch job.

        Args:
            run_id: Identifier for this batch (used as display_name)
            requests: List of request dicts from prepare_requests()
            model_name: Model to use

        Returns:
            Batch job name (use to check status and get results)
        """
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        batch_job = self.client.batches.create(
            model=model_name,
            src=requests,
            config={"display_name": run_id},
        )

        job_name = batch_job.name
        print(f"Submitted batch: {job_name}")
        print(f"  run_id: {run_id}")
        print(f"  model: {model_name}")
        print(f"  requests: {len(requests)}")
        print(f"  state: {batch_job.state.name}")

        # Save metadata if runs_dir is set
        if self.runs_dir:
            meta_path = self.runs_dir / f"{run_id}.batch_meta.json"
            meta = {
                "job_name": job_name,
                "run_id": run_id,
                "model_name": model_name,
                "num_requests": len(requests),
                "submitted_at": datetime.now().isoformat(),
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"  metadata: {meta_path.name}")

        return job_name

    def check_status(self, job_name: str) -> dict:
        """
        Check batch job status.

        Args:
            job_name: The batch job name from submit()

        Returns:
            Dict with state and progress info
        """
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

        # Print summary
        state = status["state"]
        if state == "JOB_STATE_SUCCEEDED":
            print(f"✓ SUCCEEDED: {job_name}")
        elif state == "JOB_STATE_FAILED":
            print(f"✗ FAILED: {job_name}")
        elif state == "JOB_STATE_CANCELLED":
            print(f"⊘ CANCELLED: {job_name}")
        elif state in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING"):
            label = state.replace("JOB_STATE_", "")
            print(f"⋯ {label}: {job_name}")
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
        Retrieve and parse results from a completed batch.

        Args:
            job_name: The batch job name
            chunks: Original chunks (to match with responses)

        Returns:
            List of result dicts, or None if not ready
        """
        job = self.client.batches.get(name=job_name)

        if job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"Batch not ready: {job.state.name}")
            return None

        if not hasattr(job, "dest") or not hasattr(job.dest, "inlined_responses"):
            print("No inlined_responses found.")
            return None

        responses = job.dest.inlined_responses
        print(f"Retrieved {len(responses)} responses.")

        if len(responses) != len(chunks):
            print(f"WARNING: {len(responses)} responses != {len(chunks)} chunks")

        num_chunks = len(chunks)
        results = []
        seen_indices: set[int] = set()
        index_hits = 0
        for i, resp in enumerate(responses):
            fallback_chunk = chunks[i] if i < num_chunks else {}
            fallback_id = fallback_chunk.get(
                "chunk_id",
                fallback_chunk.get("annotation_id", f"unknown_{i}"),
            )

            if resp.error:
                results.append(self._make_error_result(fallback_chunk, fallback_id, str(resp.error)))
                continue

            try:
                response_text = resp.response.text
                parsed = json.loads(response_text)

                # Match by echoed integer index (no positional fallback)
                echoed = parsed.get("chunk_id")
                if echoed is None:
                    results.append(self._make_error_result({}, f"unmatched_resp_{i}", f"Response {i}: missing chunk_id echo"))
                    continue
                try:
                    idx = int(echoed)
                except (ValueError, TypeError):
                    results.append(self._make_error_result({}, f"unmatched_resp_{i}", f"Response {i}: chunk_id '{echoed}' not an integer"))
                    continue
                if idx < 0 or idx >= num_chunks:
                    results.append(self._make_error_result({}, f"unmatched_resp_{i}", f"Response {i}: chunk_id {idx} out of range [0, {num_chunks})"))
                    continue
                if idx in seen_indices:
                    results.append(self._make_error_result({}, f"unmatched_resp_{i}", f"Response {i}: duplicate chunk_id {idx}"))
                    continue

                seen_indices.add(idx)
                index_hits += 1
                chunk = chunks[idx]
                chunk_id = chunk.get(
                    "chunk_id", chunk.get("annotation_id", f"unknown_{i}")
                )

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
                    "chunk_text": chunk["chunk_text"],
                })

            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                results.append(self._make_error_result({}, f"parse_error_{i}", f"Parse error: {e}"))

        unmatched = len(results) - index_hits
        print(f"Matched: {index_hits}/{len(responses)} | Unmatched/errors: {unmatched}")
        if unmatched:
            for r in results:
                if r.get("error"):
                    print(f"  - {r['chunk_id']}: {r['error']}")

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

    def cancel(self, job_name: str) -> None:
        """Cancel a running batch job."""
        self.client.batches.cancel(name=job_name)
        print(f"Cancelled: {job_name}")
