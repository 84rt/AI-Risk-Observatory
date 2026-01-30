"""Adoption type classifier for AIRO pipeline.

Classifies AI adoption mentions into three categories:
- non_llm: Traditional AI/ML (computer vision, predictive analytics)
- llm: Large Language Models (GPT, chatbots, text generation)
- agentic: Autonomous AI agents
"""

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from .base_classifier import BaseClassifier
from .schemas import AdoptionTypeResponse
from ..utils.prompt_loader import get_prompt_messages as render_prompt_messages


class AdoptionTypeClassifier(BaseClassifier):
    """3-category classifier for AI adoption types."""

    CLASSIFIER_TYPE = "adoption"
    RESPONSE_MODEL = AdoptionTypeResponse

    def get_prompt_messages(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Generate the classification prompts for adoption type detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")
        mention_types = metadata.get("mention_types", [])

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        return render_prompt_messages(
            "adoption_type",
            reasoning_policy="short",
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            mention_types=", ".join(mention_types) if mention_types else "unknown",
            text=text,
        )

    def extract_result(
        self, parsed: BaseModel, metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Extract classification result from AdoptionTypeResponse."""
        response: AdoptionTypeResponse = parsed  # type: ignore

        # Convert confidence scores object to dict
        adoption_confidences = response.adoption_confidences.model_dump(exclude_none=True)
        reasoning = response.reasoning or ""

        # Get active types with non-zero confidence
        active_types = [
            key for key, score in adoption_confidences.items()
            if isinstance(score, (int, float)) and score > 0
        ]
        primary_label = ",".join(sorted(active_types)) if active_types else "none"

        # Get max confidence
        confidence_scores = [
            score for score in adoption_confidences.values()
            if isinstance(score, (int, float))
        ]
        confidence = max(confidence_scores) if confidence_scores else 0.0

        return primary_label, confidence, [], reasoning

    def _legacy_parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Fallback parser for backward compatibility."""
        adoption_confidences = response.get("adoption_confidences", {})
        evidence_dict = response.get("evidence", {})
        reasoning = response.get("reasoning", "")

        if not isinstance(adoption_confidences, dict):
            adoption_confidences = {}

        active_types = [
            key for key, score in adoption_confidences.items()
            if isinstance(score, (int, float)) and score > 0
        ]
        primary_label = ",".join(sorted(active_types)) if active_types else "none"

        confidence_scores = [
            score for score in adoption_confidences.values()
            if isinstance(score, (int, float))
        ]
        confidence = max(confidence_scores) if confidence_scores else 0.0

        # Flatten evidence from dict to list
        evidence = []
        if isinstance(evidence_dict, dict):
            for category, quotes in evidence_dict.items():
                if isinstance(quotes, list):
                    for quote in quotes:
                        evidence.append(f"[{category}] {quote}")
                elif isinstance(quotes, str):
                    evidence.append(f"[{category}] {quotes}")
        elif isinstance(evidence_dict, list):
            evidence = evidence_dict

        return primary_label, confidence, evidence, reasoning
