"""Risk classifier for AIRO pipeline.

Classifies AI-related risks in company reports using the 9-category taxonomy.
Refactored from llm_classifier.py to use the base classifier pattern.
"""

from typing import Any, Dict, List, Tuple

from .base_classifier import BaseClassifier
from ..utils.prompt_loader import get_prompt_template


# Risk categories taxonomy
RISK_CATEGORIES = {
    "operational_technical": {
        "name": "Operational & Technical Risk",
        "description": "Model failures, bias, reliability, system errors, hallucinations",
    },
    "cybersecurity": {
        "name": "Cybersecurity Risk",
        "description": "AI-enabled attacks, data breaches, system vulnerabilities",
    },
    "workforce_impacts": {
        "name": "Workforce Impacts",
        "description": "Job displacement, skill obsolescence, shadow AI",
    },
    "regulatory_compliance": {
        "name": "Regulatory & Compliance Risk",
        "description": "Legal liability, AI Act, GDPR, regulatory uncertainty",
    },
    "information_integrity": {
        "name": "Information Integrity",
        "description": "Misinformation, content authenticity, deepfakes, hallucinations",
    },
    "reputational_ethical": {
        "name": "Reputational & Ethical Risk",
        "description": "Public trust, ethical concerns, bias, human rights",
    },
    "third_party_supply_chain": {
        "name": "Third-Party & Supply Chain Risk",
        "description": "Vendor reliance, downstream misuse, concentration risk",
    },
    "environmental_impact": {
        "name": "Environmental Impact",
        "description": "Energy consumption, carbon footprint, sustainability",
    },
    "national_security": {
        "name": "National Security Risk",
        "description": "Critical infrastructure, defense, systemic risk",
    },
}


class RiskClassifier(BaseClassifier):
    """9-category risk classifier for AI-related disclosures.

    Categories:
    - operational_technical
    - cybersecurity
    - workforce_impacts
    - regulatory_compliance
    - information_integrity
    - reputational_ethical
    - third_party_supply_chain
    - environmental_impact
    - national_security

    Output:
    - risk_types: List of detected risk categories
    - ai_mentioned: Whether AI is mentioned at all
    - confidence_scores: Per-category confidence
    - evidence: Quotes for each category
    - key_snippets: Key quote per category
    """

    CLASSIFIER_TYPE = "risk"

    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt for risk detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        # Build risk category descriptions
        category_descriptions = "\n".join([
            f"- **{key}**: {val['name']} - {val['description']}"
            for key, val in RISK_CATEGORIES.items()
        ])

        # Truncate text if too long
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        template = get_prompt_template("risk")
        return template.format(
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            text=text,
            risk_categories=category_descriptions,
            risk_keys=list(RISK_CATEGORIES.keys()),
        )

    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the risk classification response.

        Returns:
            Tuple of (primary_label, confidence, evidence_list, reasoning)
        """
        risk_types = response.get("risk_types", [])
        evidence_dict = response.get("evidence", {})
        key_snippets = response.get("key_snippets", {})
        confidence_scores = response.get("confidence_scores", {})
        reasoning = response.get("reasoning", "")

        # Validate risk types against known categories
        valid_risk_types = []
        for rt in risk_types:
            if rt in RISK_CATEGORIES:
                valid_risk_types.append(rt)
            else:
                self.logger.warning(f"Unknown risk category ignored: {rt}")

        # Primary label is the list of risk types
        if valid_risk_types:
            primary_label = ",".join(sorted(valid_risk_types))
        else:
            primary_label = "none"

        # Calculate average confidence
        if confidence_scores:
            valid_scores = [
                score for rt, score in confidence_scores.items()
                if rt in RISK_CATEGORIES and isinstance(score, (int, float))
            ]
            avg_confidence = sum(valid_scores) / len(valid_scores) if valid_scores else 0.5
        else:
            avg_confidence = 0.5

        # Flatten evidence from dict to list
        evidence = []
        if isinstance(evidence_dict, dict):
            for category, quotes in evidence_dict.items():
                if isinstance(quotes, list):
                    for quote in quotes:
                        evidence.append(f"[{category}] {quote}")
                elif isinstance(quotes, str) and quotes:
                    evidence.append(f"[{category}] {quotes}")
        elif isinstance(evidence_dict, list):
            evidence = evidence_dict

        # Add key snippets summary to reasoning
        if key_snippets:
            snippet_summary = "; ".join([
                f"{cat}: \"{snippet[:100]}...\""
                for cat, snippet in list(key_snippets.items())[:3]
            ])
            reasoning = f"{reasoning} | Key snippets: {snippet_summary}"

        return primary_label, avg_confidence, evidence, reasoning





