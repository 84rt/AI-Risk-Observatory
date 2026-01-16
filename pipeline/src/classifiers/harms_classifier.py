"""Binary harms classifier for AIRO pipeline.

Classifies whether a company report mentions AI-related harms.
"""

from typing import Any, Dict, List, Tuple

from .base_classifier import BaseClassifier
from ..utils.prompt_loader import get_prompt_template


class HarmsClassifier(BaseClassifier):
    """Binary classifier for AI-related harms detection.

    Answers: "Does this report mention AI-related harms?"

    Output:
    - harms_mentioned: true/false
    - confidence: 0.0-1.0
    - evidence: Up to 5 key quotes supporting classification
    """

    CLASSIFIER_TYPE = "harms"

    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt for harms detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        # Truncate text if too long (keep most relevant parts)
        max_chars = 30000
        if len(text) > max_chars:
            # Keep beginning and end, as AI mentions may be in risk sections
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        template = get_prompt_template("harms")
        return template.format(
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            text=text,
        )

    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the harms classification response.

        Returns:
            Tuple of (primary_label, confidence, evidence_list, reasoning)
        """
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





