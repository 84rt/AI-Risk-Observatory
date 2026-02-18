"""Input/output validation utilities for AIRO classifiers.

Provides validation functions for:
- Classification responses (JSON structure, valid fields)
- Company file existence and format
- Confidence score ranges
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .normalization import (
    CLASSIFIER_TYPE_ALIASES,
    RISK_SUBSTANTIVENESS_ALIASES,
    RISK_SUBSTANTIVENESS_LEVELS,
    normalize_classifier_type,
)

# Alias used throughout this module for readability in a validation context.
VALID_SUBSTANTIVENESS_LEVELS = RISK_SUBSTANTIVENESS_LEVELS

# Valid values for classification fields
VALID_CLASSIFIER_TYPES = {
    "harms",
    "adoption",
    "adoption_type",
    "substantiveness",
    "risk",
    "risk_open",
    "vendor",
    "mention_type",
    "mention_type_v2",
}

VALID_ADOPTION_TYPES = {"non_llm", "llm", "agentic"}

VALID_RISK_CATEGORIES = {
    "strategic_competitive",
    "operational_technical",
    "cybersecurity",
    "workforce_impacts",
    "regulatory_compliance",
    "information_integrity",
    "reputational_ethical",
    "third_party_supply_chain",
    "environmental_impact",
    "national_security",
    "none",
}


class ValidationError(Exception):
    """Raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(f"{field}: {message}" if field else message)


def validate_classification_response(
    response: Dict[str, Any],
    classifier_type: str,
    strict: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate a classification response from the LLM.

    Args:
        response: The parsed JSON response from the LLM
        classifier_type: Type of classifier (harms, adoption, substantiveness, risk)
        strict: If True, raise ValidationError on first failure

    Returns:
        Tuple of (is_valid, list of validation messages)
    """
    messages = []

    raw_classifier_type = classifier_type
    classifier_type = normalize_classifier_type(classifier_type)

    # Check classifier type is valid
    if raw_classifier_type not in VALID_CLASSIFIER_TYPES and classifier_type not in VALID_CLASSIFIER_TYPES:
        msg = f"Unknown classifier type: {classifier_type}"
        if strict:
            raise ValidationError(msg, "classifier_type")
        messages.append(msg)

    # Validate based on classifier type
    if classifier_type == "harms":
        is_valid, msgs = _validate_harms_response(response, strict)
        messages.extend(msgs)
    elif classifier_type == "adoption":
        is_valid, msgs = _validate_adoption_response(response, strict)
        messages.extend(msgs)
    elif classifier_type == "mention_type":
        is_valid, msgs = _validate_mention_type_response(response, strict)
        messages.extend(msgs)
    elif classifier_type == "substantiveness":
        is_valid, msgs = _validate_substantiveness_response(response, strict)
        messages.extend(msgs)
    elif classifier_type == "risk":
        is_valid, msgs = _validate_risk_response(response, strict)
        messages.extend(msgs)
    elif classifier_type == "risk_open":
        is_valid, msgs = _validate_generic_response(response, strict)
        messages.extend(msgs)
    elif classifier_type == "vendor":
        is_valid, msgs = _validate_vendor_response(response, strict)
        messages.extend(msgs)
    else:
        # Generic validation
        is_valid, msgs = _validate_generic_response(response, strict)
        messages.extend(msgs)

    return len(messages) == 0, messages


def _validate_harms_response(
    response: Dict[str, Any], strict: bool
) -> Tuple[bool, List[str]]:
    """Validate a harms classifier response."""
    messages = []

    # Required fields
    if "harms_mentioned" not in response:
        msg = "Missing required field: harms_mentioned"
        if strict:
            raise ValidationError(msg, "harms_mentioned")
        messages.append(msg)
    elif not isinstance(response["harms_mentioned"], bool):
        msg = f"harms_mentioned must be boolean, got {type(response['harms_mentioned'])}"
        if strict:
            raise ValidationError(msg, "harms_mentioned")
        messages.append(msg)

    # Validate confidence if present
    if "confidence" in response:
        conf_valid, conf_msgs = _validate_confidence(response["confidence"], strict)
        messages.extend(conf_msgs)

    # Validate evidence if present
    if "evidence" in response:
        ev_valid, ev_msgs = _validate_evidence(response["evidence"], strict)
        messages.extend(ev_msgs)

    return len(messages) == 0, messages


def _validate_adoption_response(
    response: Dict[str, Any], strict: bool
) -> Tuple[bool, List[str]]:
    """Validate an adoption type classifier response."""
    messages = []

    # Required fields
    if "adoption_signals" not in response:
        msg = "Missing required field: adoption_signals"
        if strict:
            raise ValidationError(msg, "adoption_signals")
        messages.append(msg)
    else:
        signals = response["adoption_signals"]
        if not isinstance(signals, list):
            msg = f"adoption_signals must be a list, got {type(signals)}"
            if strict:
                raise ValidationError(msg, "adoption_signals")
            messages.append(msg)
        else:
            seen = set()
            for entry in signals:
                if not isinstance(entry, dict):
                    msg = f"adoption_signals entry must be a dict, got {type(entry)}"
                    if strict:
                        raise ValidationError(msg, "adoption_signals")
                    messages.append(msg)
                    continue
                atype = entry.get("type")
                score = entry.get("signal")
                if atype not in VALID_ADOPTION_TYPES:
                    msg = f"Invalid adoption type: {atype}. Valid: {VALID_ADOPTION_TYPES}"
                    if strict:
                        raise ValidationError(msg, "adoption_signals")
                    messages.append(msg)
                else:
                    if atype in seen:
                        msg = f"Duplicate adoption type in adoption_signals: {atype}"
                        if strict:
                            raise ValidationError(msg, "adoption_signals")
                        messages.append(msg)
                    seen.add(atype)
                signal_valid, signal_msgs = _validate_signal_score(score, strict)
                messages.extend(signal_msgs)
            missing = set(VALID_ADOPTION_TYPES) - seen
            if missing:
                msg = f"adoption_signals missing required types: {', '.join(sorted(missing))}"
                if strict:
                    raise ValidationError(msg, "adoption_signals")
                messages.append(msg)

    return len(messages) == 0, messages


def _validate_substantiveness_response(
    response: Dict[str, Any], strict: bool
) -> Tuple[bool, List[str]]:
    """Validate a substantiveness classifier response."""
    messages = []

    # Required fields
    if "substantiveness" not in response:
        msg = "Missing required field: substantiveness"
        if strict:
            raise ValidationError(msg, "substantiveness")
        messages.append(msg)
    else:
        level = response["substantiveness"]
        if isinstance(level, str):
            level = RISK_SUBSTANTIVENESS_ALIASES.get(level.strip().lower(), level.strip().lower())
        if level not in VALID_SUBSTANTIVENESS_LEVELS:
            msg = f"Invalid substantiveness level: {level}. Valid: {VALID_SUBSTANTIVENESS_LEVELS}"
            if strict:
                raise ValidationError(msg, "substantiveness")
            messages.append(msg)

    # Validate confidence if present
    if "confidence" in response:
        conf_valid, conf_msgs = _validate_confidence(response["confidence"], strict)
        messages.extend(conf_msgs)

    return len(messages) == 0, messages


def _validate_risk_response(
    response: Dict[str, Any], strict: bool
) -> Tuple[bool, List[str]]:
    """Validate a risk classifier response."""
    messages = []

    # Required fields
    if "risk_types" not in response:
        msg = "Missing required field: risk_types"
        if strict:
            raise ValidationError(msg, "risk_types")
        messages.append(msg)
    else:
        types = response["risk_types"]
        if not isinstance(types, list):
            msg = f"risk_types must be a list, got {type(types)}"
            if strict:
                raise ValidationError(msg, "risk_types")
            messages.append(msg)
        else:
            for t in types:
                if t not in VALID_RISK_CATEGORIES:
                    msg = f"Invalid risk category: {t}. Valid: {VALID_RISK_CATEGORIES}"
                    if strict:
                        raise ValidationError(msg, "risk_types")
                    messages.append(msg)

    if "risk_signals" not in response:
        # Legacy support: accept confidence_scores if present
        if "confidence_scores" in response:
            conf_scores = response["confidence_scores"]
            if not isinstance(conf_scores, dict):
                msg = f"confidence_scores must be a dict, got {type(conf_scores)}"
                if strict:
                    raise ValidationError(msg, "confidence_scores")
                messages.append(msg)
            else:
                for key, value in conf_scores.items():
                    if key not in VALID_RISK_CATEGORIES:
                        msg = f"Invalid risk category in confidence_scores: {key}"
                        if strict:
                            raise ValidationError(msg, "confidence_scores")
                        messages.append(msg)
                    signal_valid, signal_msgs = _validate_signal_score(value, strict)
                    messages.extend(signal_msgs)
            # Skip risk_signals validation in legacy mode
            return len(messages) == 0, messages
        msg = "Missing required field: risk_signals"
        if strict:
            raise ValidationError(msg, "risk_signals")
        messages.append(msg)
    else:
        risk_signals = response["risk_signals"]
        if not isinstance(risk_signals, list):
            msg = f"risk_signals must be a list, got {type(risk_signals)}"
            if strict:
                raise ValidationError(msg, "risk_signals")
            messages.append(msg)
        else:
            types = response.get("risk_types", [])
            type_set = set(types) if isinstance(types, list) else set()
            seen = set()
            for entry in risk_signals:
                if not isinstance(entry, dict):
                    msg = f"risk_signals entry must be a dict, got {type(entry)}"
                    if strict:
                        raise ValidationError(msg, "risk_signals")
                    messages.append(msg)
                    continue
                key = entry.get("type")
                value = entry.get("signal")
                if key not in VALID_RISK_CATEGORIES:
                    msg = f"Invalid risk category in risk_signals: {key}"
                    if strict:
                        raise ValidationError(msg, "risk_signals")
                    messages.append(msg)
                elif key in seen:
                    msg = f"Duplicate risk type in risk_signals: {key}"
                    if strict:
                        raise ValidationError(msg, "risk_signals")
                    messages.append(msg)
                else:
                    seen.add(key)

                signal_valid, signal_msgs = _validate_signal_score(value, strict)
                messages.extend(signal_msgs)
                if isinstance(value, (int, float)) and int(value) == 0:
                    msg = "risk_signals values must be 1-3 for selected labels"
                    if strict:
                        raise ValidationError(msg, "risk_signals")
                    messages.append(msg)

            if "none" in type_set and len(type_set) > 1:
                msg = "risk_types cannot include 'none' with other labels"
                if strict:
                    raise ValidationError(msg, "risk_types")
                messages.append(msg)

            if seen != type_set:
                missing = sorted(type_set - seen)
                extra = sorted(seen - type_set)
                if missing:
                    msg = f"risk_signals missing types from risk_types: {', '.join(missing)}"
                    if strict:
                        raise ValidationError(msg, "risk_signals")
                    messages.append(msg)
                if extra:
                    msg = f"risk_signals contains types not in risk_types: {', '.join(extra)}"
                    if strict:
                        raise ValidationError(msg, "risk_signals")
                    messages.append(msg)

    # Canonical schema: categorical substantiveness
    if "substantiveness" in response:
        level = response["substantiveness"]
        if isinstance(level, str):
            level = RISK_SUBSTANTIVENESS_ALIASES.get(level.strip().lower(), level.strip().lower())
        if level not in VALID_SUBSTANTIVENESS_LEVELS:
            msg = (
                f"Invalid risk substantiveness: {response['substantiveness']}. "
                f"Valid: {VALID_SUBSTANTIVENESS_LEVELS}"
            )
            if strict:
                raise ValidationError(msg, "substantiveness")
            messages.append(msg)
    elif "substantiveness_score" in response:
        # Legacy compatibility path
        signal_valid, signal_msgs = _validate_signal_score(
            response["substantiveness_score"], strict
        )
        messages.extend(signal_msgs)
    else:
        msg = "Missing required field: substantiveness"
        if strict:
            raise ValidationError(msg, "substantiveness")
        messages.append(msg)

    return len(messages) == 0, messages


def _validate_vendor_response(
    response: Dict[str, Any], strict: bool
) -> Tuple[bool, List[str]]:
    """Validate a vendor classifier response."""
    messages = []

    if "vendors" not in response:
        msg = "Missing required field: vendors"
        if strict:
            raise ValidationError(msg, "vendors")
        messages.append(msg)
    else:
        vendors = response["vendors"]
        if not isinstance(vendors, list):
            msg = f"vendors must be a list, got {type(vendors)}"
            if strict:
                raise ValidationError(msg, "vendors")
            messages.append(msg)
        else:
            for entry in vendors:
                if not isinstance(entry, dict):
                    messages.append(f"vendor entry must be a dict, got {type(entry)}")
                elif "signal" not in entry:
                    messages.append("vendor entry missing 'signal' field")
                elif entry["signal"] not in (1, 2, 3):
                    messages.append(f"vendor signal must be 1, 2, or 3, got {entry['signal']}")

    return len(messages) == 0, messages


def _validate_mention_type_response(
    response: Dict[str, Any], strict: bool
) -> Tuple[bool, List[str]]:
    """Validate a mention type classifier response."""
    messages = []

    if "mention_types" not in response:
        msg = "Missing required field: mention_types"
        if strict:
            raise ValidationError(msg, "mention_types")
        messages.append(msg)
    else:
        if not isinstance(response["mention_types"], list):
            msg = f"mention_types must be a list, got {type(response['mention_types'])}"
            if strict:
                raise ValidationError(msg, "mention_types")
            messages.append(msg)

    if "confidence_scores" not in response:
        msg = "Missing required field: confidence_scores"
        if strict:
            raise ValidationError(msg, "confidence_scores")
        messages.append(msg)
    else:
        confidences = response["confidence_scores"]
        if not isinstance(confidences, dict):
            msg = f"confidence_scores must be a dict, got {type(confidences)}"
            if strict:
                raise ValidationError(msg, "confidence_scores")
            messages.append(msg)
        else:
            for _, score in confidences.items():
                conf_valid, conf_msgs = _validate_confidence(score, strict)
                messages.extend(conf_msgs)

    return len(messages) == 0, messages


def _validate_generic_response(
    response: Dict[str, Any], strict: bool
) -> Tuple[bool, List[str]]:
    """Generic response validation."""
    messages = []

    if not isinstance(response, dict):
        msg = f"Response must be a dict, got {type(response)}"
        if strict:
            raise ValidationError(msg, "response")
        messages.append(msg)

    return len(messages) == 0, messages


def _validate_confidence(confidence: Any, strict: bool) -> Tuple[bool, List[str]]:
    """Validate a confidence score."""
    messages = []

    if not isinstance(confidence, (int, float)):
        msg = f"confidence must be numeric, got {type(confidence)}"
        if strict:
            raise ValidationError(msg, "confidence")
        messages.append(msg)
    elif not 0.0 <= confidence <= 1.0:
        msg = f"confidence must be between 0.0 and 1.0, got {confidence}"
        if strict:
            raise ValidationError(msg, "confidence")
        messages.append(msg)

    return len(messages) == 0, messages


def _validate_signal_score(score: Any, strict: bool) -> Tuple[bool, List[str]]:
    """Validate a signal score (integer 0-3)."""
    messages = []

    if not isinstance(score, (int, float)) or isinstance(score, bool):
        msg = f"signal score must be numeric, got {type(score)}"
        if strict:
            raise ValidationError(msg, "signal_score")
        messages.append(msg)
        return len(messages) == 0, messages

    if isinstance(score, float) and not score.is_integer():
        msg = f"signal score must be an integer 0-3, got {score}"
        if strict:
            raise ValidationError(msg, "signal_score")
        messages.append(msg)
        return len(messages) == 0, messages

    score_int = int(score)
    if not 0 <= score_int <= 3:
        msg = f"signal score must be between 0 and 3, got {score}"
        if strict:
            raise ValidationError(msg, "signal_score")
        messages.append(msg)

    return len(messages) == 0, messages


def _validate_evidence(evidence: Any, strict: bool) -> Tuple[bool, List[str]]:
    """Validate evidence field."""
    messages = []

    if not isinstance(evidence, list):
        msg = f"evidence must be a list, got {type(evidence)}"
        if strict:
            raise ValidationError(msg, "evidence")
        messages.append(msg)
    else:
        for i, item in enumerate(evidence):
            if not isinstance(item, str):
                msg = f"evidence[{i}] must be a string, got {type(item)}"
                if strict:
                    raise ValidationError(msg, f"evidence[{i}]")
                messages.append(msg)

    return len(messages) == 0, messages


def validate_company_file(
    file_path: Path,
    expected_pattern: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Validate a company preprocessed file exists and is readable.

    Args:
        file_path: Path to the preprocessed markdown file
        expected_pattern: Optional regex pattern the filename should match

    Returns:
        Tuple of (is_valid, list of validation messages)
    """
    messages = []

    # Check file exists
    if not file_path.exists():
        messages.append(f"File does not exist: {file_path}")
        return False, messages

    # Check it's a file
    if not file_path.is_file():
        messages.append(f"Path is not a file: {file_path}")
        return False, messages

    # Check expected pattern
    if expected_pattern:
        if not re.match(expected_pattern, file_path.name):
            messages.append(
                f"Filename {file_path.name} does not match pattern: {expected_pattern}"
            )

    # Check file is not empty
    try:
        content = file_path.read_text(encoding="utf-8")
        if len(content.strip()) == 0:
            messages.append(f"File is empty: {file_path}")
    except Exception as e:
        messages.append(f"Error reading file {file_path}: {e}")

    return len(messages) == 0, messages


def validate_run_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a test run configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, list of validation messages)
    """
    messages = []

    required_fields = ["companies", "years", "classifiers", "model"]
    for field in required_fields:
        if field not in config:
            messages.append(f"Missing required config field: {field}")

    # Validate years
    if "years" in config:
        years = config["years"]
        if not isinstance(years, list) or len(years) == 0:
            messages.append("years must be a non-empty list")
        else:
            for year in years:
                if not isinstance(year, int) or year < 2000 or year > 2100:
                    messages.append(f"Invalid year: {year}")

    # Validate classifiers
    if "classifiers" in config:
        classifiers = config["classifiers"]
        if not isinstance(classifiers, list) or len(classifiers) == 0:
            messages.append("classifiers must be a non-empty list")
        else:
            for clf in classifiers:
                canonical = CLASSIFIER_TYPE_ALIASES.get(clf, clf)
                if clf not in VALID_CLASSIFIER_TYPES and canonical not in VALID_CLASSIFIER_TYPES:
                    messages.append(f"Invalid classifier: {clf}")

    return len(messages) == 0, messages


def parse_json_response(response_text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse a JSON response from the LLM, handling common issues.

    Args:
        response_text: Raw response text from the LLM

    Returns:
        Tuple of (parsed_dict or None, error_message or None)
    """
    # Try direct JSON parse
    try:
        return json.loads(response_text), None
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    json_text = response_text

    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end > start:
            json_text = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        if end > start:
            json_text = response_text[start:end].strip()

    # Try parsing the extracted text
    try:
        return json.loads(json_text), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
