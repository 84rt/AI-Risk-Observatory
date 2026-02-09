"""Classifier modules for AIRO pipeline.

This module contains all classifiers for analyzing AI-related disclosures
in company annual reports.
"""

from .base_classifier import BaseClassifier, ClassificationResult, BatchResult
from .harms_classifier import HarmsClassifier
from .adoption_type_classifier import AdoptionTypeClassifier
from .mention_type_classifier import MentionTypeClassifier
from .substantiveness_classifier import SubstantivenessClassifier
from .vendor_classifier import VendorClassifier
from .risk_classifier import RiskClassifier
from .llm_classifier_v2 import LLMClassifierV2
from .schemas import (
    MentionType,
    MentionConfidenceScores,
    MentionTypeResponse,
    MentionTypeResponseV2,
    MentionTypeResponseV2,
    AdoptionType,
    AdoptionConfidenceScores,
    AdoptionTypeResponse,
    RiskType,
    RiskConfidenceScores,
    RiskResponse,
    VendorTag,
    VendorEntry,
    VendorResponse,
    SubstantivenessLevel,
    SubstantivenessScores,
    SubstantivenessResponse,
    HarmsResponse,
)

__all__ = [
    # Base classes
    "BaseClassifier",
    "ClassificationResult",
    "BatchResult",
    # Classifiers
    "HarmsClassifier",
    "AdoptionTypeClassifier",
    "MentionTypeClassifier",
    "LLMClassifierV2",
    "SubstantivenessClassifier",
    "VendorClassifier",
    "RiskClassifier",
    # Schemas - Enums
    "MentionType",
    "AdoptionType",
    "RiskType",
    "VendorTag",
    "SubstantivenessLevel",
    # Schemas - Confidence score models
    "MentionConfidenceScores",
    "AdoptionConfidenceScores",
    "RiskConfidenceScores",
    "VendorEntry",
    "SubstantivenessScores",
    # Schemas - Response models
    "MentionTypeResponse",
    "MentionTypeResponseV2",
    "AdoptionTypeResponse",
    "RiskResponse",
    "VendorResponse",
    "SubstantivenessResponse",
    "HarmsResponse",
]
