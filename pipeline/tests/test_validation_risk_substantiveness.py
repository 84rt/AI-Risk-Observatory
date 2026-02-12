"""Regression tests for risk substantiveness validation compatibility."""

from src.utils.validation import validate_classification_response


def _base_risk_payload() -> dict:
    return {
        "risk_types": ["cybersecurity"],
        "risk_signals": [{"type": "cybersecurity", "signal": 3}],
    }


def test_risk_validation_accepts_canonical_substantiveness() -> None:
    payload = _base_risk_payload()
    payload["substantiveness"] = "moderate"
    ok, msgs = validate_classification_response(payload, "risk")
    assert ok, msgs


def test_risk_validation_accepts_legacy_alias_contextual() -> None:
    payload = _base_risk_payload()
    payload["substantiveness"] = "contextual"
    ok, msgs = validate_classification_response(payload, "risk")
    assert ok, msgs


def test_risk_validation_accepts_legacy_substantiveness_score() -> None:
    payload = _base_risk_payload()
    payload["substantiveness_score"] = 2
    ok, msgs = validate_classification_response(payload, "risk")
    assert ok, msgs


def test_risk_validation_rejects_missing_substantiveness() -> None:
    payload = _base_risk_payload()
    ok, msgs = validate_classification_response(payload, "risk")
    assert not ok
    assert any("Missing required field: substantiveness" in msg for msg in msgs)
