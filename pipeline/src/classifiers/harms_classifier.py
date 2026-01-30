"""Binary harms classifier for AIRO pipeline.

Classifies whether a company report mentions AI-related harms.
"""

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from .base_classifier import BaseClassifier
from .schemas import HarmsResponse
from ..utils.prompt_loader import get_prompt_messages as render_prompt_messages


class HarmsClassifier(BaseClassifier):
    """Binary classifier for AI-related harms detection."""

    CLASSIFIER_TYPE = "harms"
    RESPONSE_MODEL = HarmsResponse

    def get_prompt_messages(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Generate the classification prompts for harms detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        return render_prompt_messages(
            "harms",
            reasoning_policy="short",
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            text=text,
        )

    def extract_result(
        self, parsed: BaseModel, metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Extract classification result from HarmsResponse."""
        response: HarmsResponse = parsed  # type: ignore

        harms_mentioned = response.harms_mentioned
        confidence = response.confidence
        evidence = response.evidence or []
        reasoning = response.reasoning or ""

        # Ensure evidence is a list
        if isinstance(evidence, str):
            evidence = [evidence] if evidence else []
        elif not isinstance(evidence, list):
            evidence = []

        # Primary label is the boolean value as string
        primary_label = "true" if harms_mentioned else "false"

        return primary_label, confidence, evidence, reasoning

    def _legacy_parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Fallback parser for backward compatibility."""
        harms_mentioned = response.get("harms_mentioned", False)
        confidence = response.get("confidence", 0.5)
        evidence = response.get("evidence", [])
        reasoning = response.get("reasoning", "")

        # Ensure evidence is a list
        if isinstance(evidence, str):
            evidence = [evidence] if evidence else []
        elif not isinstance(evidence, list):
            evidence = []

        # Primary label is the boolean value as string
        primary_label = "true" if harms_mentioned else "false"

        return primary_label, confidence, evidence, reasoning
