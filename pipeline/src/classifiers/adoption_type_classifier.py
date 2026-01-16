"""Adoption type classifier for AIRO pipeline.

Classifies AI adoption mentions into three categories:
- non_llm: Traditional AI/ML (computer vision, predictive analytics)
- llm: Large Language Models (GPT, chatbots, text generation)
- agentic: Autonomous AI agents
"""

from typing import Any, Dict, List, Tuple

from .base_classifier import BaseClassifier
from ..utils.prompt_loader import get_prompt_template


class AdoptionTypeClassifier(BaseClassifier):
    """3-category classifier for AI adoption types.

    Categories:
    - non_llm: Traditional AI/ML (computer vision, predictive analytics,
               fraud detection, recommendation systems)
    - llm: Large Language Models (GPT, BERT, chatbots, text generation, NLP)
    - agentic: Agentic AI (autonomous agents, self-directed systems)

    Output:
    - adoption_confidences: Per-type confidence scores
    - evidence: Quotes for each type
    """

    CLASSIFIER_TYPE = "adoption"

    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt for adoption type detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        # Truncate text if too long
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]
        template = get_prompt_template("adoption_type")
        return template.format(
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            text=text,
        )

    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the adoption type classification response.

        Returns:
            Tuple of (primary_label, confidence, evidence_list, reasoning)
        """
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
