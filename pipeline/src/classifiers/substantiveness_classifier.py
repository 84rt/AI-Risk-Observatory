"""Substantiveness classifier for AIRO pipeline.

Classifies AI mentions as boilerplate, contextual, or substantive.
"""

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from .base_classifier import BaseClassifier
from .schemas import SubstantivenessResponse
from ..utils.prompt_loader import get_prompt_messages as render_prompt_messages


class SubstantivenessClassifier(BaseClassifier):
    """Substantiveness classifier for AI disclosures."""

    CLASSIFIER_TYPE = "substantiveness"
    RESPONSE_MODEL = SubstantivenessResponse

    def get_prompt_messages(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Generate the classification prompts for substantiveness detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")
        mention_types = metadata.get("mention_types", [])

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        return render_prompt_messages(
            "substantiveness",
            reasoning_policy=self.reasoning_policy,
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            mention_types=", ".join(mention_types) if mention_types else "unknown",
            text=text,
        )

    def extract_result(
        self, parsed: BaseModel, metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Extract classification result from SubstantivenessResponse."""
        response: SubstantivenessResponse = parsed  # type: ignore

        # Convert confidence scores object to dict
        substantiveness_scores = response.substantiveness_scores.model_dump(exclude_none=True)
        reasoning = response.reasoning or ""

        # Determine primary label from highest score
        valid_levels = {"boilerplate", "contextual", "substantive"}
        best_level = "boilerplate"
        best_score = 0.0

        for level, score in substantiveness_scores.items():
            if level in valid_levels and isinstance(score, (int, float)) and score > best_score:
                best_level = level
                best_score = score

        primary_label = best_level
        confidence = best_score if best_score > 0 else 0.5

        return primary_label, confidence, [], reasoning

    def _legacy_parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Fallback parser for backward compatibility."""
        # Handle old format with single "substantiveness" field
        substantiveness = response.get("substantiveness", "boilerplate")
        confidence = response.get("confidence", 0.5)
        evidence_dict = response.get("evidence", {})
        reasoning = response.get("reasoning", "")
        substantive_ratio = response.get("substantive_ratio", 0.0)
        indicators = response.get("indicators", [])

        # Also handle new format with scores
        substantiveness_scores = response.get("substantiveness_scores", {})
        if substantiveness_scores:
            valid_levels = {"boilerplate", "contextual", "substantive"}
            best_level = "boilerplate"
            best_score = 0.0
            for level, score in substantiveness_scores.items():
                if level in valid_levels and isinstance(score, (int, float)) and score > best_score:
                    best_level = level
                    best_score = score
            substantiveness = best_level
            confidence = best_score if best_score > 0 else 0.5

        # Validate substantiveness value
        valid_levels = {"boilerplate", "contextual", "substantive"}
        if substantiveness not in valid_levels:
            substantiveness = "boilerplate"

        primary_label = substantiveness

        # Flatten evidence from dict to list
        evidence = []
        if isinstance(evidence_dict, dict):
            for level, quotes in evidence_dict.items():
                if isinstance(quotes, list):
                    for quote in quotes:
                        evidence.append(f"[{level}] {quote}")
                elif isinstance(quotes, str) and quotes:
                    evidence.append(f"[{level}] {quotes}")
        elif isinstance(evidence_dict, list):
            evidence = evidence_dict

        # Add ratio and indicators to reasoning
        reasoning_parts = [reasoning]
        if substantive_ratio:
            reasoning_parts.append(f"Substantive ratio: {substantive_ratio:.0%}")
        if indicators:
            reasoning_parts.append(f"Indicators: {', '.join(indicators[:5])}")

        full_reasoning = " | ".join(filter(None, reasoning_parts))

        return primary_label, confidence, evidence, full_reasoning
