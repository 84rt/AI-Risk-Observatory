"""Mention type classifier for AIRO pipeline.

Assigns multi-label mention types with confidence scores:
- adoption
- risk
- harm
- vendor
- general_ambiguous
"""

from typing import Any, Dict, List, Tuple

from .base_classifier import BaseClassifier
from ..utils.prompt_loader import get_prompt_template


class MentionTypeClassifier(BaseClassifier):
    """Multi-label mention type classifier."""

    CLASSIFIER_TYPE = "mention_type"

    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt for mention typing."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")
        report_section = metadata.get("report_section", "Unknown")

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        template = get_prompt_template("mention_type")
        return template.format(
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            report_section=report_section,
            text=text,
        )

    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the mention type classification response."""
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
