"""Regression tests for classifier prompt/schema compatibility."""

from pydantic import ValidationError

from src.classifiers.schemas import AdoptionTypeResponse, MentionTypeResponseV2
from src.utils.prompt_loader import get_prompt_messages
from src.utils.validation import validate_classification_response


def test_prompt_loader_accepts_current_and_legacy_keys() -> None:
    kwargs = {
        "firm_name": "Example Plc",
        "sector": "Technology",
        "report_year": "2024",
        "report_section": "Risk",
        "mention_types": "adoption",
        "text": "We use AI tools to improve operations.",
    }
    for key in ("mention_type", "mention_type_v3", "adoption", "adoption_type", "risk", "risk_v5"):
        system, user = get_prompt_messages(key, reasoning_policy="short", **kwargs)
        assert "Example Plc" in system
        assert "{reasoning_instruction}" not in system
        assert "We use AI tools" in user


def test_adoption_schema_and_validation_accept_ambiguous() -> None:
    payload = {
        "adoption_signals": [
            {"type": "non_llm", "signal": 0},
            {"type": "llm", "signal": 0},
            {"type": "agentic", "signal": 0},
            {"type": "ambiguous", "signal": 2},
        ],
        "substantiveness": "moderate",
    }
    AdoptionTypeResponse.model_validate(payload)
    ok, msgs = validate_classification_response(payload, "adoption_type")
    assert ok, msgs


def test_adoption_ambiguous_does_not_cooccur_with_specific_types() -> None:
    payload = {
        "adoption_signals": [
            {"type": "non_llm", "signal": 1},
            {"type": "llm", "signal": 0},
            {"type": "agentic", "signal": 0},
            {"type": "ambiguous", "signal": 2},
        ],
        "substantiveness": "moderate",
    }
    try:
        AdoptionTypeResponse.model_validate(payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("ambiguous should not co-occur with specific adoption types")

    ok, msgs = validate_classification_response(payload, "adoption_type")
    assert not ok
    assert any("ambiguous adoption signal must not co-occur" in msg for msg in msgs)


def test_mention_schema_accepts_general_ambiguous_alias() -> None:
    parsed = MentionTypeResponseV2.model_validate(
        {
            "mention_types": ["general_ambiguous"],
            "confidence_scores": {"general_ambiguous": 0.8},
        }
    )
    assert [label.value for label in parsed.mention_types] == ["general_other_or_ambiguous"]
    assert parsed.confidence_scores.general_other_or_ambiguous == 0.8
