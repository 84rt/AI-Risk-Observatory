"""Risk classifier for AIRO pipeline.

Classifies AI-related risks in company reports using the 10-category taxonomy.
"""

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from .base_classifier import BaseClassifier
from .schemas import RiskResponse, RiskType
from ..utils.prompt_loader import get_prompt_messages as render_prompt_messages


# Risk categories taxonomy (for reference and validation)
RISK_CATEGORIES = {
    "strategic_market": {
        "name": "Strategic & Market Risk",
        "description": "Failure to adopt, market displacement, competitive disadvantage",
    },
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
    """10-category risk classifier for AI-related disclosures."""

    CLASSIFIER_TYPE = "risk"
    RESPONSE_MODEL = RiskResponse

    def get_prompt_messages(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Generate the classification prompts for risk detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")
        report_section = metadata.get("report_section", "Unknown")
        mention_types = metadata.get("mention_types", [])

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        return render_prompt_messages(
            "risk",
            reasoning_policy=self.reasoning_policy,
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            report_section=report_section,
            mention_types=", ".join(mention_types) if mention_types else "unknown",
            text=text,
        )

    def extract_result(
        self, parsed: BaseModel, metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Extract classification result from RiskResponse."""
        response: RiskResponse = parsed  # type: ignore

        # Convert enum values to strings
        risk_types = [rt.value for rt in response.risk_types]
        # Convert confidence scores object to dict
        confidence_scores = response.confidence_scores.model_dump(exclude_none=True)
        reasoning = response.reasoning or ""

        # Validate risk types against known categories
        valid_risk_types = [
            rt for rt in risk_types
            if rt in RISK_CATEGORIES or rt == "none"
        ]

        # Primary label is the list of risk types
        if valid_risk_types and valid_risk_types != ["none"]:
            primary_label = ",".join(sorted([rt for rt in valid_risk_types if rt != "none"]))
        else:
            primary_label = "none"

        # Calculate average confidence for valid categories
        if confidence_scores:
            valid_scores = [
                score for rt, score in confidence_scores.items()
                if (rt in RISK_CATEGORIES or rt == "none") and isinstance(score, (int, float))
            ]
            avg_confidence = sum(valid_scores) / len(valid_scores) if valid_scores else 0.5
        else:
            avg_confidence = 0.5

        return primary_label, avg_confidence, [], reasoning

    def _legacy_parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Fallback parser for backward compatibility."""
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
