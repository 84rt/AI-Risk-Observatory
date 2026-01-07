"""Abstract base class for all AIRO classifiers.

Provides common functionality for:
- Structured logging
- Result validation
- Evidence extraction
- Database storage
- Retry logic
- Batch processing
"""

import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..utils.logging_config import (
    ClassifierLogAdapter,
    get_classifier_logger,
    log_api_call,
    log_classification_result,
    log_classification_start,
    log_error,
)
from ..utils.validation import parse_json_response, validate_classification_response

settings = get_settings()


@dataclass
class ClassificationResult:
    """Result from a single classification."""

    # Identifiers
    result_id: str
    run_id: str
    firm_id: str
    firm_name: str
    report_year: int
    classifier_type: str

    # Classification output
    classification: Dict[str, Any]
    primary_label: str
    confidence_score: float

    # Evidence
    evidence: List[str] = field(default_factory=list)
    key_snippet: Optional[str] = None

    # Traceability
    source_file: str = ""
    prompt_hash: str = ""
    response_raw: str = ""
    reasoning: str = ""

    # Timing and tokens
    api_latency_ms: int = 0
    tokens_used: int = 0

    # Status
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/export."""
        return {
            "result_id": self.result_id,
            "run_id": self.run_id,
            "firm_id": self.firm_id,
            "firm_name": self.firm_name,
            "report_year": self.report_year,
            "classifier_type": self.classifier_type,
            "classification": self.classification,
            "primary_label": self.primary_label,
            "confidence_score": self.confidence_score,
            "evidence": self.evidence,
            "key_snippet": self.key_snippet,
            "source_file": self.source_file,
            "prompt_hash": self.prompt_hash,
            "response_raw": self.response_raw,
            "reasoning": self.reasoning,
            "api_latency_ms": self.api_latency_ms,
            "tokens_used": self.tokens_used,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class BatchResult:
    """Result from batch classification."""

    run_id: str
    classifier_type: str
    total_items: int
    success_count: int
    error_count: int
    results: List[ClassificationResult]
    avg_confidence: float
    low_confidence_count: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/export."""
        return {
            "run_id": self.run_id,
            "classifier_type": self.classifier_type,
            "total_items": self.total_items,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "avg_confidence": self.avg_confidence,
            "low_confidence_count": self.low_confidence_count,
            "duration_seconds": self.duration_seconds,
            "results": [r.to_dict() for r in self.results],
        }


class BaseClassifier(ABC):
    """Abstract base class for all classifiers.

    Provides common functionality for LLM-based classification:
    - Gemini API integration
    - Structured logging
    - Result validation
    - Evidence extraction
    - Retry logic
    """

    # Class-level constants - override in subclasses
    CLASSIFIER_TYPE: str = "base"
    LOW_CONFIDENCE_THRESHOLD: float = 0.7

    def __init__(
        self,
        run_id: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Initialize the classifier.

        Args:
            run_id: Unique identifier for this test run
            api_key: Gemini API key (defaults to settings)
            model_name: Model to use (defaults to settings)
            temperature: LLM temperature (0.0 for deterministic)
        """
        self.run_id = run_id
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self.temperature = temperature

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize model with JSON response mode
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": settings.max_tokens,
            "response_mime_type": "application/json",
        }

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
        )

        # Set up logging
        self.logger = get_classifier_logger(
            self.CLASSIFIER_TYPE,
            self.run_id,
            create_separate_file=True,
        )

    @abstractmethod
    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt.

        Args:
            text: The text to classify
            metadata: Additional context (firm_name, year, sector, etc.)

        Returns:
            The prompt string to send to the LLM
        """
        pass

    @abstractmethod
    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the LLM response into structured result.

        Args:
            response: Parsed JSON response from LLM
            metadata: Original metadata

        Returns:
            Tuple of (primary_label, confidence, evidence_list, reasoning)
        """
        pass

    def classify(
        self,
        text: str,
        metadata: Dict[str, Any],
        source_file: str = "",
    ) -> ClassificationResult:
        """Classify a single text excerpt.

        Args:
            text: The text to classify
            metadata: Context dict with firm_id, firm_name, report_year, etc.
            source_file: Path to source file for traceability

        Returns:
            ClassificationResult with full details
        """
        firm_id = metadata.get("firm_id", "unknown")
        firm_name = metadata.get("firm_name", firm_id)
        report_year = metadata.get("report_year", 0)

        # Log start
        log_classification_start(self.logger, firm_id, report_year, source_file)

        # Generate result ID
        result_id = f"{self.run_id}_{self.CLASSIFIER_TYPE}_{firm_id}_{report_year}_{uuid.uuid4().hex[:8]}"

        # Build prompt
        prompt = self.get_prompt(text, metadata)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        # Call LLM with retry
        start_time = time.time()
        try:
            response_text, tokens = self._call_llm(prompt)
            latency_ms = int((time.time() - start_time) * 1000)

            log_api_call(
                self.logger,
                self.model_name,
                len(prompt) // 4,  # Approximate prompt tokens
                tokens,
                latency_ms,
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            log_error(self.logger, firm_id, type(e).__name__, str(e))

            return ClassificationResult(
                result_id=result_id,
                run_id=self.run_id,
                firm_id=firm_id,
                firm_name=firm_name,
                report_year=report_year,
                classifier_type=self.CLASSIFIER_TYPE,
                classification={},
                primary_label="error",
                confidence_score=0.0,
                source_file=source_file,
                prompt_hash=prompt_hash,
                api_latency_ms=latency_ms,
                success=False,
                error_message=str(e),
            )

        # Parse response
        parsed, error = parse_json_response(response_text)
        if error or parsed is None:
            log_error(self.logger, firm_id, "ParseError", error or "Empty response")

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
                response_raw=response_text,
                api_latency_ms=latency_ms,
                tokens_used=tokens,
                success=False,
                error_message=error,
            )

        # Validate response
        is_valid, messages = validate_classification_response(
            parsed, self.CLASSIFIER_TYPE
        )
        if not is_valid:
            self.logger.warning(f"Validation warnings: {messages}")

        # Parse into structured result
        try:
            primary_label, confidence, evidence, reasoning = self.parse_result(
                parsed, metadata
            )
        except Exception as e:
            log_error(self.logger, firm_id, "ParseResultError", str(e))
            primary_label = "error"
            confidence = 0.0
            evidence = []
            reasoning = f"Error parsing result: {e}"

        # Determine key snippet
        key_snippet = evidence[0] if evidence else None

        # Log result
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
            key_snippet=key_snippet,
            source_file=source_file,
            prompt_hash=prompt_hash,
            response_raw=response_text,
            reasoning=reasoning,
            api_latency_ms=latency_ms,
            tokens_used=tokens,
            success=True,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _call_llm(self, prompt: str) -> Tuple[str, int]:
        """Call the LLM with retry logic.

        Args:
            prompt: The prompt to send

        Returns:
            Tuple of (response_text, token_count)
        """
        response = self.model.generate_content(prompt)
        text = response.text

        # Estimate tokens from response
        tokens = len(text) // 4

        return text, tokens

    def run_batch(
        self,
        items: List[Dict[str, Any]],
        text_key: str = "text",
        rate_limit_delay: float = 0.5,
    ) -> BatchResult:
        """Process multiple items with logging and storage.

        Args:
            items: List of dicts with text and metadata
            text_key: Key in item dict containing the text to classify
            rate_limit_delay: Delay between API calls in seconds

        Returns:
            BatchResult with all classification results
        """
        start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        confidences = []

        self.logger.info(f"Starting batch classification of {len(items)} items")

        for i, item in enumerate(items, 1):
            text = item.get(text_key, "")
            metadata = {k: v for k, v in item.items() if k != text_key}
            source_file = item.get("source_file", "")

            result = self.classify(text, metadata, source_file)
            results.append(result)

            if result.success:
                success_count += 1
                confidences.append(result.confidence_score)
            else:
                error_count += 1

            # Progress logging
            if i % 5 == 0 or i == len(items):
                self.logger.info(f"Progress: {i}/{len(items)} classified")

            # Rate limiting
            if i < len(items):
                time.sleep(rate_limit_delay)

        duration = time.time() - start_time
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        low_conf_count = sum(1 for c in confidences if c < self.LOW_CONFIDENCE_THRESHOLD)

        self.logger.info(
            f"Batch complete: {success_count}/{len(items)} successful, "
            f"avg_confidence={avg_confidence:.2f}, "
            f"low_confidence={low_conf_count}, "
            f"duration={duration:.1f}s"
        )

        return BatchResult(
            run_id=self.run_id,
            classifier_type=self.CLASSIFIER_TYPE,
            total_items=len(items),
            success_count=success_count,
            error_count=error_count,
            results=results,
            avg_confidence=avg_confidence,
            low_confidence_count=low_conf_count,
            duration_seconds=duration,
        )

    def classify_report(
        self,
        report_path: Path,
        firm_id: str,
        firm_name: str,
        report_year: int,
        sector: str = "Unknown",
    ) -> ClassificationResult:
        """Classify a full preprocessed report file.

        Args:
            report_path: Path to the preprocessed markdown file
            firm_id: Company identifier
            firm_name: Company name
            report_year: Year of the report
            sector: Company sector

        Returns:
            ClassificationResult
        """
        # Read report content
        text = report_path.read_text(encoding="utf-8")

        metadata = {
            "firm_id": firm_id,
            "firm_name": firm_name,
            "report_year": report_year,
            "sector": sector,
        }

        return self.classify(text, metadata, str(report_path))





