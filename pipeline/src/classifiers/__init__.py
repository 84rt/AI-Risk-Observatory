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
from .schemas import (
    MentionType,
    MentionConfidenceScores,
    MentionTypeResponse,
    AdoptionType,
    AdoptionConfidenceScores,
    AdoptionTypeResponse,
    RiskType,
    RiskConfidenceScores,
    RiskResponse,
    VendorTag,
    VendorConfidenceScores,
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
    "VendorConfidenceScores",
    "SubstantivenessScores",
    # Schemas - Response models
    "MentionTypeResponse",
    "AdoptionTypeResponse",
    "RiskResponse",
    "VendorResponse",
    "SubstantivenessResponse",
    "HarmsResponse",
]
