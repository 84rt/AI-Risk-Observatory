"""LLM-based classification for AI risk mentions using Google Gemini."""

import json
import logging
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .chunker import CandidateSpan
from .utils.prompt_loader import get_prompt_template

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Classification:
    """LLM classification result for a text span."""

    # Relevance
    is_relevant: bool

    # If not relevant, remaining fields are None
    mention_type: Optional[str] = None
    ai_specificity: Optional[str] = None
    frontier_tech_flag: Optional[bool] = None

    # Risk classification
    tier_1_category: Optional[str] = None
    tier_2_driver: Optional[str] = None

    # Specificity and materiality
    specificity_level: Optional[str] = None
    materiality_signal: Optional[str] = None

    # Governance
    mitigation_mentioned: Optional[bool] = None
    governance_maturity: Optional[str] = None

    # Metadata
    confidence_score: Optional[float] = None
    reasoning_summary: Optional[str] = None


class LLMClassifier:
    """Classifier using Google Gemini for AI risk detection."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the classifier.

        Args:
            api_key: Gemini API key. If not provided, uses settings.
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = settings.gemini_model

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize model with JSON response mode
        generation_config = {
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_tokens,
            "response_mime_type": "application/json",
        }

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=settings.retry_delay, max=30)
    )
    def classify_span(
        self,
        candidate: CandidateSpan
    ) -> Classification:
        """Classify a candidate span for AI relevance and risk.

        Args:
            candidate: CandidateSpan to classify

        Returns:
            Classification result
        """
        logger.debug(f"Classifying span {candidate.span_id}")

        # Build the prompt
        prompt = self._build_classification_prompt(candidate)

        # Call Gemini
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
        except Exception as e:
            logger.error(f"Gemini API error for {candidate.span_id}: {e}")
            # Return not relevant if API fails
            return Classification(is_relevant=False)

        # Parse response
        try:
            classification_data = self._parse_response(response_text)
            classification = Classification(**classification_data)
            logger.debug(
                f"Span {candidate.span_id}: "
                f"relevant={classification.is_relevant}"
            )
            return classification
        except Exception as e:
            logger.error(
                f"Failed to parse classification response for {candidate.span_id}: {e}"
            )
            logger.error(f"Response was: {response_text}")
            # Return not relevant if parsing fails
            return Classification(is_relevant=False)

    def _build_classification_prompt(self, candidate: CandidateSpan) -> str:
        """Build the classification prompt.

        Args:
            candidate: CandidateSpan to classify

        Returns:
            Prompt string
        """
        template = get_prompt_template("legacy_mention_classifier")
        return template.format(
            firm_name=candidate.firm_name,
            sector=candidate.sector,
            report_year=candidate.report_year,
            report_section=candidate.report_section or "Unknown",
            text=candidate.text,
        )

    @staticmethod
    def _parse_response(response_text: str) -> dict:
        """Parse the JSON response from Gemini.

        Args:
            response_text: Response text from Gemini

        Returns:
            Dict with classification data
        """
        # Gemini with JSON mode should return clean JSON
        # But handle markdown code blocks just in case
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text.strip()

        # Parse JSON
        data = json.loads(json_text)

        # Validate is_relevant is present
        if "is_relevant" not in data:
            raise ValueError("Response missing 'is_relevant' field")

        return data


def classify_candidates(
    candidates: list[CandidateSpan],
    batch_size: int = 10
) -> list[tuple[CandidateSpan, Classification]]:
    """Classify a batch of candidate spans.

    Args:
        candidates: List of CandidateSpan objects
        batch_size: Number to process before logging progress

    Returns:
        List of (candidate, classification) tuples
    """
    classifier = LLMClassifier()
    results = []

    total = len(candidates)
    for i, candidate in enumerate(candidates, 1):
        if i % batch_size == 0:
            logger.info(f"Classified {i}/{total} spans...")

        classification = classifier.classify_span(candidate)
        results.append((candidate, classification))

    logger.info(f"Completed classification of {total} spans")

    return results
