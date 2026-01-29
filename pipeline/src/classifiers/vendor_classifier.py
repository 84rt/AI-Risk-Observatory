"""Vendor classifier for AIRO pipeline.

Extracts AI vendor/provider mentions from company reports.
"""

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from .base_classifier import BaseClassifier
from .schemas import VendorResponse
from ..utils.prompt_loader import get_prompt_messages as render_prompt_messages


class VendorClassifier(BaseClassifier):
    """AI vendor/provider extraction classifier."""

    CLASSIFIER_TYPE = "vendor"
    RESPONSE_MODEL = VendorResponse

    def get_prompt_messages(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Generate the classification prompts for vendor extraction."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        return render_prompt_messages(
            "vendor",
            reasoning_policy=self.reasoning_policy,
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            text=text,
        )

    def extract_result(
        self, parsed: BaseModel, metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Extract classification result from VendorResponse."""
        response: VendorResponse = parsed  # type: ignore

        # Convert confidence scores object to dict
        vendor_confidences = response.vendor_confidences.model_dump(exclude_none=True)
        other_vendor = response.other_vendor
        reasoning = response.reasoning or ""

        # Get active vendors with non-zero confidence
        active_vendors = [
            key for key, score in vendor_confidences.items()
            if isinstance(score, (int, float)) and score > 0
        ]
        if other_vendor:
            active_vendors.append(f"other:{other_vendor}")
        primary_label = ",".join(sorted(active_vendors)) if active_vendors else "none"

        # Get max confidence
        confidence_scores = [
            score for score in vendor_confidences.values()
            if isinstance(score, (int, float))
        ]
        confidence = max(confidence_scores) if confidence_scores else 0.0

        return primary_label, confidence, [], reasoning

    def _legacy_parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Fallback parser for backward compatibility."""
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
