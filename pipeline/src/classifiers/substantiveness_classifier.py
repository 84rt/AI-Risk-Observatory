"""Substantiveness classifier for AIRO pipeline.

Classifies AI mentions as boilerplate, contextual, or substantive.
"""

from typing import Any, Dict, List, Tuple

from .base_classifier import BaseClassifier
from ..utils.prompt_loader import get_prompt_template


class SubstantivenessClassifier(BaseClassifier):
    """Substantiveness classifier for AI disclosures.

    Categories:
    - boilerplate: Generic legal phrasing applicable to any company
    - contextual: Sector-relevant but non-specific
    - substantive: Named systems, quantified impact, concrete mitigations

    Output:
    - substantiveness: Primary classification
    - confidence: 0.0-1.0
    - evidence: Quotes demonstrating the substantiveness level
    - substantive_ratio: Approximate % of substantive content
    """

    CLASSIFIER_TYPE = "substantiveness"

    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt for substantiveness detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        # Truncate text if too long
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        template = get_prompt_template("substantiveness")
        return template.format(
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            text=text,
        )

    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the substantiveness classification response.

        Returns:
            Tuple of (primary_label, confidence, evidence_list, reasoning)
        """
        substantiveness = response.get("substantiveness", "boilerplate")
        confidence = response.get("confidence", 0.5)
        evidence_dict = response.get("evidence", {})
        reasoning = response.get("reasoning", "")
        substantive_ratio = response.get("substantive_ratio", 0.0)
        indicators = response.get("indicators", [])

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





