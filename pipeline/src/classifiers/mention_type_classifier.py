"""Mention type classifier for AIRO pipeline.

Assigns multi-label mention types with confidence scores:
- adoption
- risk
- harm
- vendor
- general_ambiguous
- none
"""

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from .base_classifier import BaseClassifier
from .schemas import MentionTypeResponseV2
from ..utils.prompt_loader import get_prompt_messages as render_prompt_messages


class MentionTypeClassifier(BaseClassifier):
    """Multi-label mention type classifier."""

    CLASSIFIER_TYPE = "mention_type"
    RESPONSE_MODEL = MentionTypeResponseV2
    PROMPT_KEY = "mention_type_v3"
    SCHEMA_VERSION = "mention_type_v2"

    def get_prompt_messages(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Generate the classification prompts for mention typing."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")
        report_section = metadata.get("report_section", "Unknown")

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        return render_prompt_messages(
            self.PROMPT_KEY,
            reasoning_policy="short",  # Always include brief reasoning in output
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            report_section=report_section,
            text=text,
        )

    def extract_result(
        self, parsed: BaseModel, metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Extract classification result from MentionTypeResponseV2."""
        response: MentionTypeResponseV2 = parsed  # type: ignore

        # Convert enum values to strings
        mention_types = [mt.value for mt in response.mention_types]
        # Convert confidence scores object to dict
        confidence_scores = response.confidence_scores.model_dump(exclude_none=True)
        reasoning = response.reasoning or ""

        primary_label = ",".join(sorted(mention_types)) if mention_types else "none"

        # Get max confidence
        valid_scores = [
            score for score in confidence_scores.values()
            if isinstance(score, (int, float))
        ]
        confidence = max(valid_scores) if valid_scores else 0.0

        return primary_label, confidence, [], reasoning

    def _legacy_parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Fallback parser for backward compatibility."""
        mention_types = response.get("mention_types", [])
        confidence_scores = response.get("confidence_scores", {})
        reasoning = response.get("reasoning", "")

        if isinstance(mention_types, str):
            mention_types = [mention_types]
        if not isinstance(mention_types, list):
            mention_types = []
        if not isinstance(confidence_scores, dict):
            confidence_scores = {}

        active_types = [
            tag for tag, score in confidence_scores.items()
            if isinstance(score, (int, float)) and score > 0
        ]
        primary_label = ",".join(sorted(active_types)) if active_types else "none"

        valid_scores = [
            score for score in confidence_scores.values()
            if isinstance(score, (int, float))
        ]
        confidence = max(valid_scores) if valid_scores else 0.0

        return primary_label, confidence, [], reasoning
