"""Vendor classifier for AIRO pipeline.

Extracts AI vendor/provider mentions from company reports.
"""

from typing import Any, Dict, List, Tuple

from .base_classifier import BaseClassifier
from ..utils.prompt_loader import get_prompt_template


class VendorClassifier(BaseClassifier):
    """AI vendor/provider extraction classifier.

    Extracts mentions of AI vendors and products from reports.

    Target Vendors:
    - OpenAI (GPT, ChatGPT, DALL-E)
    - Microsoft (Azure OpenAI, Copilot, Azure ML)
    - Google (Gemini, Vertex AI, Google Cloud AI)
    - Amazon (Bedrock, SageMaker, AWS AI)
    - Anthropic (Claude)
    - Meta (LLaMA)
    - Internal/Custom solutions
    - Other vendors

    Output:
    - vendor_confidences: Per-vendor confidence scores
    - evidence: Quotes for each vendor mention
    """

    CLASSIFIER_TYPE = "vendor"

    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt for vendor extraction."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        # Truncate text if too long
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        template = get_prompt_template("vendor")
        return template.format(
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            text=text,
        )

    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the vendor extraction response.

        Returns:
            Tuple of (primary_label, confidence, evidence_list, reasoning)
        """
        vendor_confidences = response.get("vendor_confidences", {})
        other_vendor = response.get("other_vendor", "")
        reasoning = response.get("reasoning", "")

        if not isinstance(vendor_confidences, dict):
            vendor_confidences = {}

        active_vendors = [
            key for key, score in vendor_confidences.items()
            if isinstance(score, (int, float)) and score > 0
        ]
        if other_vendor:
            active_vendors.append(f"other:{other_vendor}")
        primary_label = ",".join(sorted(active_vendors)) if active_vendors else "none"

        confidence_scores = [
            score for score in vendor_confidences.values()
            if isinstance(score, (int, float))
        ]
        confidence = max(confidence_scores) if confidence_scores else 0.0

        # Extract evidence from vendors
        evidence = []
        evidence_dict = response.get("evidence", {})
        if isinstance(evidence_dict, dict):
            for vendor, quotes in evidence_dict.items():
                if isinstance(quotes, list):
                    for quote in quotes:
                        evidence.append(f"[{vendor}] {quote}")
                elif isinstance(quotes, str) and quotes:
                    evidence.append(f"[{vendor}] {quotes}")

        return primary_label, confidence, evidence, reasoning




