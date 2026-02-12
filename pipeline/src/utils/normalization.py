"""Shared normalization helpers for classifier payloads and labels.

This module is intentionally compatibility-first: it keeps legacy aliases and
fallback parsing paths so existing runs remain readable while new runs use the
canonical schema.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

RISK_LABEL_ALIASES: Dict[str, str] = {
    "strategic_market": "strategic_competitive",
    "regulatory": "regulatory_compliance",
    "workforce": "workforce_impacts",
    "environmental": "environmental_impact",
}

RISK_SUBSTANTIVENESS_LEVELS = {"boilerplate", "moderate", "substantive"}
RISK_SUBSTANTIVENESS_ALIASES = {"contextual": "moderate"}

CLASSIFIER_TYPE_ALIASES: Dict[str, str] = {
    "adoption_type": "adoption",
    "mention_type_v2": "mention_type",
}


def normalize_classifier_type(value: str) -> str:
    return CLASSIFIER_TYPE_ALIASES.get(value, value)


def normalize_risk_label(value: str) -> str:
    token = str(value).strip()
    return RISK_LABEL_ALIASES.get(token, token)


def normalize_risk_labels(values: Iterable[str]) -> list[str]:
    return [normalize_risk_label(v) for v in values]


def normalize_risk_substantiveness(value: Any) -> Optional[str]:
    """Normalize risk substantiveness to canonical categorical labels.

    Accepted canonical values:
    - boilerplate
    - moderate
    - substantive

    Legacy compatible inputs:
    - "contextual" -> "moderate"
    - numeric 0-1 bucketized by thirds
    - numeric 0-3 rounded then mapped to categories
    """
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if score <= 1.0:
            if score >= 0.67:
                return "substantive"
            if score >= 0.34:
                return "moderate"
            return "boilerplate"
        rounded = int(round(score))
        if rounded >= 3:
            return "substantive"
        if rounded == 2:
            return "moderate"
        return "boilerplate"

    token = str(value).strip().lower()
    token = RISK_SUBSTANTIVENESS_ALIASES.get(token, token)
    if token in RISK_SUBSTANTIVENESS_LEVELS:
        return token
    return None


def normalize_signal_to_unit_interval(score: float, max_signal: float = 3.0) -> float:
    """Convert signal-like numeric score to [0, 1] range."""
    if not isinstance(score, (int, float)):
        return 0.0
    if max_signal <= 0:
        return 0.0
    value = float(score) / float(max_signal)
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def signals_list_to_map(value: Any) -> Dict[str, float]:
    """Convert [{type, signal}] or {type: signal} into a normalized map."""
    if isinstance(value, dict):
        return {
            str(k): float(v)
            for k, v in value.items()
            if isinstance(v, (int, float))
        }
    if isinstance(value, list):
        out: Dict[str, float] = {}
        for entry in value:
            if not isinstance(entry, dict):
                continue
            key = entry.get("type") or entry.get("label")
            sig = entry.get("signal")
            if key is not None and isinstance(sig, (int, float)):
                out[str(key)] = float(sig)
        return out
    return {}


def risk_signals_from_payload(payload: dict) -> Dict[str, float]:
    """Extract risk signal map from canonical or legacy risk payloads."""
    if not isinstance(payload, dict):
        return {}
    signals = payload.get("risk_signals")
    if signals is not None:
        return signals_list_to_map(signals)
    legacy = payload.get("confidence_scores")
    if legacy is not None:
        return signals_list_to_map(legacy)
    return {}

