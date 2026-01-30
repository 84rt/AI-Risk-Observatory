"""Abstract base class for all AIRO classifiers.

Provides common functionality for:
- Structured logging
- Result validation
- Evidence extraction
- Database storage
- Retry logic
- Batch processing with tqdm progress bars
- Pydantic schema validation for LLM responses
"""

import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from ..config import get_settings
from ..utils.logging_config import (
    ClassifierLogAdapter,
    get_classifier_logger,
    log_api_call,
    log_classification_result,
    log_classification_start,
    log_error,
    log_llm_request_response,
    save_debug_log,
)
from ..utils.validation import parse_json_response, validate_classification_response

settings = get_settings()

# Type variable for Pydantic response models
T = TypeVar("T", bound=BaseModel)


def _clean_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Pydantic JSON schema to Gemini-compatible format.

    Gemini's response_schema supports a subset of JSON Schema:
    - type, format, description, nullable, enum, items, properties, required

    Does NOT support:
    - $defs/$ref (references), additionalProperties, anyOf/oneOf/allOf,
      title, default, examples
    """
    # Store definitions for inlining $ref
    defs = schema.get("$defs", schema.get("definitions", {}))

    def resolve_ref(ref: str) -> Dict[str, Any]:
        """Resolve a $ref to its definition."""
        if ref.startswith("#/$defs/"):
            def_name = ref.split("/")[-1]
            if def_name in defs:
                return clean_recursive(defs[def_name])
        elif ref.startswith("#/definitions/"):
            def_name = ref.split("/")[-1]
            if def_name in defs:
                return clean_recursive(defs[def_name])
        return {}

    def clean_recursive(obj: Any) -> Any:
        if not isinstance(obj, dict):
            if isinstance(obj, list):
                return [clean_recursive(item) for item in obj]
            return obj

        # Handle $ref - inline the definition
        if "$ref" in obj:
            resolved = resolve_ref(obj["$ref"])
            # Merge with any other properties in obj (except $ref)
            for k, v in obj.items():
                if k != "$ref" and k not in resolved:
                    resolved[k] = clean_recursive(v)
            return resolved

        # Handle anyOf for nullable types
        if "anyOf" in obj:
            any_of = obj["anyOf"]
            non_null = [item for item in any_of if item.get("type") != "null"]
            if len(non_null) == 1:
                # It's a nullable type
                result = clean_recursive(non_null[0])
                result["nullable"] = True
                # Preserve description if present
                if "description" in obj:
                    result["description"] = obj["description"]
                return result
            # Multiple non-null types - take the first one
            if non_null:
                return clean_recursive(non_null[0])
            return {"type": "STRING", "nullable": True}

        # Fields to keep
        supported_keys = {
            "type", "format", "description", "nullable", "enum",
            "items", "properties", "required"
        }

        cleaned = {}

        for k, v in obj.items():
            if k not in supported_keys:
                continue

            if k == "type":
                # Convert JSON Schema types to Gemini types (uppercase)
                type_map = {
                    "string": "STRING",
                    "number": "NUMBER",
                    "integer": "INTEGER",
                    "boolean": "BOOLEAN",
                    "array": "ARRAY",
                    "object": "OBJECT",
                    "null": "STRING",  # Gemini doesn't have null type
                }
                cleaned[k] = type_map.get(v, v.upper() if isinstance(v, str) else v)
            elif k == "items":
                cleaned[k] = clean_recursive(v)
            elif k == "properties":
                cleaned[k] = {pk: clean_recursive(pv) for pk, pv in v.items()}
            elif k == "enum":
                cleaned[k] = v  # Keep enum values as-is
            else:
                cleaned[k] = v

        return cleaned

    return clean_recursive(schema)


def _clean_schema_for_openrouter(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Pydantic JSON schema to OpenRouter-compatible format.

    Simplifies the schema by:
    - Inlining $defs/$ref references
    - Converting anyOf nullable unions to simple types
    - Removing unsupported fields (title, default, examples)

    Keeps lowercase type names (standard JSON Schema format) unlike Gemini.
    """
    defs = schema.get("$defs", schema.get("definitions", {}))

    def resolve_ref(ref: str) -> Dict[str, Any]:
        """Resolve a $ref to its definition."""
        if ref.startswith("#/$defs/"):
            def_name = ref.split("/")[-1]
            if def_name in defs:
                return clean_recursive(defs[def_name])
        elif ref.startswith("#/definitions/"):
            def_name = ref.split("/")[-1]
            if def_name in defs:
                return clean_recursive(defs[def_name])
        return {}

    def clean_recursive(obj: Any) -> Any:
        if not isinstance(obj, dict):
            if isinstance(obj, list):
                return [clean_recursive(item) for item in obj]
            return obj

        # Handle $ref - inline the definition
        if "$ref" in obj:
            resolved = resolve_ref(obj["$ref"])
            for k, v in obj.items():
                if k != "$ref" and k not in resolved:
                    resolved[k] = clean_recursive(v)
            return resolved

        # Handle anyOf for nullable types (common in Pydantic Optional fields)
        if "anyOf" in obj:
            any_of = obj["anyOf"]
            non_null = [item for item in any_of if item.get("type") != "null"]
            if len(non_null) == 1:
                result = clean_recursive(non_null[0])
                # Preserve description if present
                if "description" in obj:
                    result["description"] = obj["description"]
                return result
            if non_null:
                return clean_recursive(non_null[0])
            return {"type": "string"}

        # Fields to keep (standard JSON Schema)
        supported_keys = {
            "type", "format", "description", "enum",
            "items", "properties", "required"
        }

        cleaned = {}
        for k, v in obj.items():
            if k not in supported_keys:
                continue

            if k == "type":
                cleaned[k] = v  # Keep lowercase
            elif k == "items":
                cleaned[k] = clean_recursive(v)
            elif k == "properties":
                cleaned[k] = {pk: clean_recursive(pv) for pk, pv in v.items()}
            elif k == "enum":
                cleaned[k] = v
            else:
                cleaned[k] = v

        return cleaned

    return clean_recursive(schema)


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
    - Gemini API integration with native structured output
    - OpenRouter fallback with schema injection
    - Pydantic validation for responses
    - Structured logging
    - Retry logic
    - tqdm progress bars for batch processing
    """

    # Class-level constants - override in subclasses
    CLASSIFIER_TYPE: str = "base"
    LOW_CONFIDENCE_THRESHOLD: float = 0.7
    RESPONSE_MODEL: Optional[Type[BaseModel]] = None  # Pydantic model for response

    def __init__(
        self,
        run_id: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        use_openrouter: Optional[bool] = None,
        thinking_budget: int = 0,
    ):
        """Initialize the classifier.

        Args:
            run_id: Unique identifier for this test run
            api_key: Gemini API key (defaults to settings)
            model_name: Model to use (defaults to settings)
            temperature: LLM temperature (0.0 for deterministic)
            thinking_budget: Token budget for model thinking/reasoning.
                0 = disabled, >0 = token budget (e.g., 1024, 4096, 8192).
                Only supported by models with native thinking (Gemini 2.0+).
        """
        self.run_id = run_id
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.use_openrouter = (
            use_openrouter
            if use_openrouter is not None
            else bool(settings.openrouter_api_key and "/" in self.model_name)
        )

        self.gemini_client = None
        if not self.use_openrouter:
            from google import genai

            # Create Gemini client with new API
            self.gemini_client = genai.Client(api_key=self.api_key)

        # Set up logging
        self.logger = get_classifier_logger(
            self.CLASSIFIER_TYPE,
            self.run_id,
            create_separate_file=True,
        )

    @abstractmethod
    def get_prompt_messages(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Generate the classification prompts.

        Args:
            text: The text to classify
            metadata: Additional context (firm_name, year, sector, etc.)

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        pass

    def parse_response(self, response_text: str) -> BaseModel:
        """Parse and validate LLM response using Pydantic model.

        Args:
            response_text: Raw JSON response from LLM

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationError: If response doesn't match schema
            json.JSONDecodeError: If response isn't valid JSON
        """
        if self.RESPONSE_MODEL is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define RESPONSE_MODEL"
            )

        data = json.loads(response_text)
        return self.RESPONSE_MODEL.model_validate(data)

    @abstractmethod
    def extract_result(
        self, parsed: BaseModel, metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Extract classification result from validated Pydantic model.

        Args:
            parsed: Validated Pydantic response model
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

        # Build prompts
        system_prompt, user_prompt = self.get_prompt_messages(text, metadata)
        combined_prompt = (
            f"SYSTEM:\\n{system_prompt}\\n\\nUSER:\\n{user_prompt}"
            if system_prompt
            else user_prompt
        )
        prompt_hash = hashlib.sha256(combined_prompt.encode()).hexdigest()[:16]

        # Call LLM with retry
        start_time = time.time()
        try:
            response_text, tokens = self._call_llm(system_prompt, user_prompt)
            latency_ms = int((time.time() - start_time) * 1000)

            prompt_chars = len(combined_prompt)
            response_chars = len(response_text)
            log_api_call(
                self.logger,
                self.model_name,
                prompt_chars // 4,  # Approximate prompt tokens
                response_chars // 4,
                latency_ms,
                prompt_chars=prompt_chars,
                response_chars=response_chars,
            )

            # Log full request/response for debugging
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

            # Save detailed debug log to file
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
                },
                response_data={
                    "response_text": response_text,
                    "tokens": tokens,
                    "latency_ms": latency_ms,
                },
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            log_error(self.logger, firm_id, type(e).__name__, str(e))

            # Log failed request for debugging
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

        # Parse and validate response using Pydantic
        try:
            parsed_model = self.parse_response(response_text)
            parsed = parsed_model.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            # Fall back to raw JSON parsing for backward compatibility
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
                    error_message=str(e),
                )
            # Use fallback parsed dict
            parsed_model = None

        # Validate response structure (legacy validation)
        is_valid, messages = validate_classification_response(
            parsed, self.CLASSIFIER_TYPE
        )
        if not is_valid:
            self.logger.warning(f"Validation warnings: {messages}")

        # Extract structured result
        try:
            if parsed_model is not None:
                primary_label, confidence, evidence, reasoning = self.extract_result(
                    parsed_model, metadata
                )
            else:
                # Fallback to legacy parse_result for backward compatibility
                primary_label, confidence, evidence, reasoning = self._legacy_parse_result(
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

    def _legacy_parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Fallback to legacy parse_result if Pydantic parsing fails.

        Override in subclasses for backward compatibility.
        """
        # Default implementation returns empty result
        return "unknown", 0.0, [], ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Tuple[str, int]:
        """Call the LLM with retry logic.

        Args:
            system_prompt: System prompt content
            user_prompt: User prompt content

        Returns:
            Tuple of (response_text, token_count)
        """
        if self.use_openrouter:
            text = self._call_openrouter(system_prompt, user_prompt)
        else:
            text = self._call_gemini(system_prompt, user_prompt)

        # Estimate tokens from response
        tokens = len(text) // 4

        return text, tokens

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Call Gemini API using the new google.genai SDK.

        Supports native thinking for compatible models (gemini-2.5+, gemini-3).
        """
        from google.genai import types

        # Build generation config
        config_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": settings.max_tokens,
            "response_mime_type": "application/json",
        }

        # Add response_schema if RESPONSE_MODEL is defined
        if self.RESPONSE_MODEL is not None:
            raw_schema = self.RESPONSE_MODEL.model_json_schema()
            clean_schema = _clean_schema_for_gemini(raw_schema)
            config_kwargs["response_schema"] = clean_schema

        # Add thinking config if budget > 0
        if self.thinking_budget > 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        config = types.GenerateContentConfig(**config_kwargs)

        # Build contents with system instruction
        contents = []
        if system_prompt:
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}")]
            ))
        else:
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=user_prompt)]
            ))

        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        return response.text

    def _call_openrouter(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenRouter API with schema injection.

        Since OpenRouter doesn't support native response_schema,
        we inject a cleaned/simplified schema into the system prompt.
        """
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        # Inject schema instruction for OpenRouter
        enhanced_system = system_prompt
        if self.RESPONSE_MODEL is not None:
            # Clean the schema to remove $defs, $ref, anyOf for better compatibility
            raw_schema = self.RESPONSE_MODEL.model_json_schema()
            clean_schema = _clean_schema_for_openrouter(raw_schema)
            schema_json = json.dumps(clean_schema, indent=2)
            schema_instruction = f"""

## OUTPUT FORMAT
Return JSON matching this schema:
{schema_json}

Return ONLY valid JSON."""
            enhanced_system = system_prompt + schema_instruction

        url = f"{settings.openrouter_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if enhanced_system:
            messages.append({"role": "system", "content": enhanced_system})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": settings.max_tokens,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter error {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def run_batch(
        self,
        items: List[Dict[str, Any]],
        text_key: str = "text",
        rate_limit_delay: float = 0.5,
        show_progress: bool = True,
    ) -> BatchResult:
        """Process multiple items with tqdm progress bar.

        Args:
            items: List of dicts with text and metadata
            text_key: Key in item dict containing the text to classify
            rate_limit_delay: Delay between API calls in seconds
            show_progress: Whether to show tqdm progress bar

        Returns:
            BatchResult with all classification results
        """
        start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        confidences = []

        self.logger.info(f"Starting batch classification of {len(items)} items")

        # Create tqdm progress bar
        iterator = (
            tqdm(items, desc=self.CLASSIFIER_TYPE, unit="item")
            if show_progress
            else items
        )

        for item in iterator:
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

            # Update progress bar postfix with stats
            if show_progress and hasattr(iterator, "set_postfix"):
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                iterator.set_postfix(
                    success=success_count,
                    errors=error_count,
                    avg_conf=f"{avg_conf:.2f}",
                )

            # Rate limiting
            if rate_limit_delay > 0:
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
