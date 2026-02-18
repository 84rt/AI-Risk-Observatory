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
from .open_risk_classifier import OpenRiskDiscoveryClassifier
from .llm_classifier_v2 import LLMClassifierV2
from .schemas import (
    MentionType,
    MentionConfidenceScores,
    MentionTypeResponse,
    MentionTypeResponseV2,
    AdoptionType,
    AdoptionSignalEntry,
    AdoptionTypeResponse,
    RiskType,
    RiskConfidenceScores,
    RiskResponse,
    OpenRiskSignalEntry,
    OpenRiskLabelDefinitionEntry,
    OpenRiskEvidenceEntry,
    OpenRiskResponse,
    VendorTag,
    VendorEntry,
    VendorResponse,
    SubstantivenessLevel,
    SubstantivenessScores,
    SubstantivenessResponse,
    SubstantivenessResponseV2,
    HarmsResponse,
)

# Backward compatibility for older imports that still expect this symbol name.
AdoptionConfidenceScores = AdoptionSignalEntry

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
    "OpenRiskDiscoveryClassifier",
    # Schemas - Enums
    "MentionType",
    "AdoptionType",
    "RiskType",
    "VendorTag",
    "SubstantivenessLevel",
    # Schemas - Confidence score models
    "MentionConfidenceScores",
    "AdoptionSignalEntry",
    "AdoptionConfidenceScores",
    "RiskConfidenceScores",
    "OpenRiskSignalEntry",
    "OpenRiskLabelDefinitionEntry",
    "OpenRiskEvidenceEntry",
    "VendorEntry",
    "SubstantivenessScores",
    # Schemas - Response models
    "MentionTypeResponse",
    "MentionTypeResponseV2",
    "AdoptionTypeResponse",
    "RiskResponse",
    "OpenRiskResponse",
    "VendorResponse",
    "SubstantivenessResponse",
    "SubstantivenessResponseV2",
    "HarmsResponse",
]
