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
        mention_types = metadata.get("mention_types", [])

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        return render_prompt_messages(
            "vendor",
            reasoning_policy="short",
            firm_name=firm_name,
            sector=sector,
            report_year=report_year,
            mention_types=", ".join(mention_types) if mention_types else "unknown",
            text=text,
        )

    def extract_result(
        self, parsed: BaseModel, metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Extract classification result from VendorResponse."""
        response: VendorResponse = parsed  # type: ignore

        vendor_confidences = response.vendor_confidences.model_dump(exclude_none=True)
        reasoning = response.reasoning or ""

        # Get active vendors with non-zero confidence
        active_vendors = [
            key for key, score in vendor_confidences.items()
            if isinstance(score, (int, float)) and score > 0
        ]
        if response.other_vendor:
            active_vendors.append(f"other:{response.other_vendor}")
        primary_label = ",".join(sorted(active_vendors)) if active_vendors else "none"

        # Max confidence across all vendors
        scores = [
            s for s in vendor_confidences.values()
            if isinstance(s, (int, float))
        ]
        confidence = max(scores) if scores else 0.0

        return primary_label, confidence, [], reasoning
