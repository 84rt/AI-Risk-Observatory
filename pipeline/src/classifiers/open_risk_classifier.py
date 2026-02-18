"""Conservative open-taxonomy risk discovery classifier for AIRO pipeline.

Reuses the epistemic stance of the fixed-taxonomy risk classifier while allowing
emergent risk labels when supported by the excerpt.
"""

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from .base_classifier import BaseClassifier
from .schemas import OpenRiskResponse
from ..utils.prompt_loader import get_prompt_messages as render_prompt_messages


class OpenRiskDiscoveryClassifier(BaseClassifier):
    """Open-taxonomy AI risk discovery classifier."""

    CLASSIFIER_TYPE = "risk_open"
    RESPONSE_MODEL = OpenRiskResponse
    PROMPT_KEY = "risk_open_v1"
    SCHEMA_VERSION = "risk_open_v1"

    def get_prompt_messages(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Generate prompts for conservative open-risk discovery."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")
        report_section = metadata.get("report_section", "Unknown")
        mention_types = metadata.get("mention_types", [])

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        return render_prompt_messages(
            self.PROMPT_KEY,
            reasoning_policy="short",
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
        """Extract normalized labels, confidence, evidence, and reasoning."""
        response: OpenRiskResponse = parsed  # type: ignore

        risk_types = [str(rt) for rt in response.risk_types]
        reasoning = response.reasoning or ""

        if set(risk_types) == {"none"}:
            primary_label = "none"
        else:
            primary_label = ",".join(sorted(rt for rt in risk_types if rt != "none"))

        signals = {
            str(entry.type): int(entry.signal)
            for entry in response.risk_signals
        }

        if primary_label == "none":
            confidence = float(signals.get("none", 1)) / 3.0
        else:
            active_scores = [
                score
                for label, score in signals.items()
                if label != "none" and isinstance(score, (int, float))
            ]
            confidence = (sum(active_scores) / len(active_scores) / 3.0) if active_scores else 0.0

        evidence = [
            f"[{entry.type}] {entry.snippet}"
            for entry in response.evidence
            if entry.snippet
        ]

        return primary_label, confidence, evidence, reasoning

    def _legacy_parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Fallback parser for backward compatibility."""
        risk_types_raw = response.get("risk_types", [])
        risk_types = [str(rt).strip().lower() for rt in risk_types_raw if rt is not None]

        if set(risk_types) == {"none"}:
            primary_label = "none"
        else:
            primary_label = ",".join(sorted(rt for rt in risk_types if rt and rt != "none"))
            if not primary_label:
                primary_label = "none"

        risk_signals = response.get("risk_signals", [])
        signal_map: Dict[str, int] = {}
        if isinstance(risk_signals, list):
            for entry in risk_signals:
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get("type", "")).strip().lower()
                val = entry.get("signal")
                if key and isinstance(val, (int, float)):
                    signal_map[key] = int(val)

        if primary_label == "none":
            confidence = float(signal_map.get("none", 1)) / 3.0
        else:
            active_scores = [
                score
                for key, score in signal_map.items()
                if key != "none" and isinstance(score, (int, float))
            ]
            confidence = (sum(active_scores) / len(active_scores) / 3.0) if active_scores else 0.0

        evidence = []
        raw_evidence = response.get("evidence", [])
        if isinstance(raw_evidence, list):
            for entry in raw_evidence:
                if isinstance(entry, dict):
                    label = str(entry.get("type", "")).strip().lower()
                    snippet = str(entry.get("snippet", "")).strip()
                    if label and snippet:
                        evidence.append(f"[{label}] {snippet}")

        reasoning = str(response.get("reasoning", ""))
        return primary_label, confidence, evidence, reasoning
