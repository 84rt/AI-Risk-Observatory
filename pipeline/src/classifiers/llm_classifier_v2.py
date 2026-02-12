"""Stricter mention-type classifier with structured output and parse retry."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError

from .base_classifier import BaseClassifier, ClassificationResult
from .schemas import MentionTypeResponseV2
from ..utils.logging_config import (
    log_api_call,
    log_classification_result,
    log_classification_start,
    log_error,
    log_llm_request_response,
    save_debug_log,
)
from ..utils.validation import validate_classification_response
from ..utils.prompt_loader import get_prompt_messages as render_prompt_messages


class LLMClassifierV2(BaseClassifier):
    """Mention-type classifier with stricter schema enforcement and retry."""

    CLASSIFIER_TYPE = "mention_type_v2"
    RESPONSE_MODEL = MentionTypeResponseV2
    PROMPT_KEY = "mention_type_v3"
    SCHEMA_VERSION = "mention_type_v2"

    def __init__(
        self,
        run_id: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        use_openrouter: Optional[bool] = None,
        thinking_budget: int = 0,
        parse_retries: int = 2,
    ):
        super().__init__(
            run_id=run_id,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            use_openrouter=use_openrouter,
            thinking_budget=thinking_budget,
        )
        self.parse_retries = parse_retries

    def get_prompt_messages(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")
        report_section = metadata.get("report_section", "Unknown")

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        return render_prompt_messages(
            self.PROMPT_KEY,
            reasoning_policy="short",
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            report_section=report_section,
            text=text,
        )

    def extract_result(
        self, parsed: BaseModel, metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        response: MentionTypeResponseV2 = parsed  # type: ignore

        mention_types = [mt.value for mt in response.mention_types]
        confidence_scores = response.confidence_scores.model_dump(exclude_none=True)
        reasoning = response.reasoning or ""

        active_types = list(mention_types)
        primary_label = ",".join(sorted(active_types)) if active_types else "none"

        valid_scores = [
            score for score in confidence_scores.values()
            if isinstance(score, (int, float))
        ]
        confidence = max(valid_scores) if valid_scores else 0.0

        return primary_label, confidence, [], reasoning

    def classify(
        self,
        text: str,
        metadata: Dict[str, Any],
        source_file: str = "",
    ) -> ClassificationResult:
        firm_id = metadata.get("firm_id", "unknown")
        firm_name = metadata.get("firm_name", firm_id)
        report_year = metadata.get("report_year", 0)

        log_classification_start(self.logger, firm_id, report_year, source_file)

        result_id = f"{self.run_id}_{self.CLASSIFIER_TYPE}_{firm_id}_{report_year}_{uuid.uuid4().hex[:8]}"

        system_prompt, user_prompt = self.get_prompt_messages(text, metadata)
        combined_prompt = (
            f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
            if system_prompt
            else user_prompt
        )
        prompt_hash = hashlib.sha256(combined_prompt.encode()).hexdigest()[:16]

        last_error: Optional[str] = None
        for attempt in range(self.parse_retries + 1):
            start_time = time.time()
            try:
                response_text, tokens = self._call_llm(system_prompt, user_prompt)
                latency_ms = int((time.time() - start_time) * 1000)

                prompt_chars = len(combined_prompt)
                response_chars = len(response_text)
                log_api_call(
                    self.logger,
                    self.model_name,
                    prompt_chars // 4,
                    response_chars // 4,
                    latency_ms,
                    prompt_chars=prompt_chars,
                    response_chars=response_chars,
                )

                log_llm_request_response(
                    self.logger,
                    firm_id,
                    self.model_name,
                    system_prompt,
                    user_prompt,
                    response_text,
                    latency_ms,
                    success=True,
                )

                save_debug_log(
                    run_id=self.run_id,
                    classifier_type=self.CLASSIFIER_TYPE,
                    firm_id=firm_id,
                    request_data={
                        "model": self.model_name,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "prompt_hash": prompt_hash,
                        "metadata": metadata,
                        "attempt": attempt,
                    },
                    response_data={
                        "response_text": response_text,
                        "tokens": tokens,
                        "latency_ms": latency_ms,
                    },
                )

                parsed_model = self.parse_response(response_text)
                parsed = parsed_model.model_dump()

                is_valid, messages = validate_classification_response(
                    parsed, "mention_type"
                )
                if not is_valid:
                    self.logger.warning(f"Validation warnings: {messages}")

                primary_label, confidence, evidence, reasoning = self.extract_result(
                    parsed_model, metadata
                )

                log_classification_result(
                    self.logger,
                    firm_id,
                    primary_label,
                    confidence,
                    len(evidence),
                    latency_ms,
                )

                return ClassificationResult(
                    result_id=result_id,
                    run_id=self.run_id,
                    firm_id=firm_id,
                    firm_name=firm_name,
                    report_year=report_year,
                    classifier_type=self.CLASSIFIER_TYPE,
                    classification=parsed,
                    primary_label=primary_label,
                    confidence_score=confidence,
                    evidence=evidence,
                    key_snippet=evidence[0] if evidence else None,
                    source_file=source_file,
                    prompt_hash=prompt_hash,
                    response_raw=response_text,
                    reasoning=reasoning,
                    api_latency_ms=latency_ms,
                    tokens_used=tokens,
                    success=True,
                )

            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = str(e)
                if attempt < self.parse_retries:
                    # Strengthen the system prompt for retry
                    system_prompt = (
                        system_prompt
                        + "\n\nIMPORTANT: Return ONLY valid JSON that matches the schema. "
                        "Do not include any extra text."
                    )
                    continue

            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                log_error(self.logger, firm_id, type(e).__name__, str(e))
                log_llm_request_response(
                    self.logger,
                    firm_id,
                    self.model_name,
                    system_prompt,
                    user_prompt,
                    "",
                    latency_ms,
                    success=False,
                    error_message=str(e),
                )
                last_error = str(e)
                break

        return ClassificationResult(
            result_id=result_id,
            run_id=self.run_id,
            firm_id=firm_id,
            firm_name=firm_name,
            report_year=report_year,
            classifier_type=self.CLASSIFIER_TYPE,
            classification={},
            primary_label="parse_error",
            confidence_score=0.0,
            source_file=source_file,
            prompt_hash=prompt_hash,
            response_raw="",
            reasoning="",
            api_latency_ms=0,
            tokens_used=0,
            success=False,
            error_message=last_error or "parse_error",
        )
