"""Pydantic response schemas for LLM classifiers.

These schemas serve dual purposes:
1. Type validation for LLM responses
2. Field descriptions provide context to Gemini's response_schema

Note: Dict types are avoided for Gemini compatibility. Instead, we use
nested objects with explicit properties for known keys.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Mention Type Classifier
# =============================================================================


class MentionType(str, Enum):
    """Types of AI mentions in company reports."""

    adoption = "adoption"
    risk = "risk"
    harm = "harm"
    vendor = "vendor"
    general_ambiguous = "general_ambiguous"
    none = "none"


class MentionConfidenceScores(BaseModel):
    """Confidence scores for each mention type."""

    adoption: Optional[float] = Field(
        default=None,
        description="Confidence for adoption mention (0.0-1.0)",
    )
    risk: Optional[float] = Field(
        default=None,
        description="Confidence for risk mention (0.0-1.0)",
    )
    harm: Optional[float] = Field(
        default=None,
        description="Confidence for harm mention (0.0-1.0)",
    )
    vendor: Optional[float] = Field(
        default=None,
        description="Confidence for vendor mention (0.0-1.0)",
    )
    general_ambiguous: Optional[float] = Field(
        default=None,
        description="Confidence for general/ambiguous mention (0.0-1.0)",
    )
    none: Optional[float] = Field(
        default=None,
        description="Confidence for no AI mention / false positive (0.0-1.0)",
    )


class MentionTypeResponse(BaseModel):
    """Response schema for mention type classification."""

    mention_types: List[MentionType] = Field(
        description=(
            "Detected mention types (non-mutually exclusive). "
            "'adoption': AI deployment/implementation by company or clients. "
            "'risk': AI described as risk, downside, or material concern. "
            "'harm': AI causing or enabling harm. "
            "'vendor': Named AI vendor/platform mentioned. "
            "'general_ambiguous': Vague AI reference, high-level plans. "
            "'none': False positive, no AI mention."
        )
    )
    confidence_scores: MentionConfidenceScores = Field(
        description=(
            "Confidence per detected type (0.0-1.0). "
            "0.0=no evidence, 0.2=implied, 0.5=plausible, 0.8=explicit, 0.95=unambiguous. "
            "Only set scores for types listed in mention_types."
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale.",
    )


class MentionTypeResponseV2(BaseModel):
    """Stricter response schema for mention type classification."""

    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale.",
    )
    mention_types: List[MentionType] = Field(
        description=(
            "Detected mention types (non-mutually exclusive). "
            "'adoption': AI deployment/implementation by company or clients. "
            "'risk': AI described as risk, downside, or material concern. "
            "'harm': AI causing or enabling harm. "
            "'vendor': Named AI vendor/platform mentioned. "
            "'general_ambiguous': Vague AI reference, high-level plans. "
            "'none': False positive, no AI mention."
        )
    )
    confidence_scores: MentionConfidenceScores = Field(
        description=(
            "Confidence per detected type (0.0-1.0). "
            "Only include scores for types listed in mention_types, "
            "and include a score for every type in mention_types."
        )
    )

    @model_validator(mode="after")
    def validate_confidences(self) -> "MentionTypeResponseV2":
        types = {mt.value for mt in self.mention_types}
        if not types:
            raise ValueError("mention_types must not be empty")
        if "none" in types and len(types) > 1:
            raise ValueError("mention_types cannot include 'none' with other labels")

        scores = self.confidence_scores.model_dump()
        for label, score in scores.items():
            if score is not None and label not in types:
                raise ValueError(
                    f"confidence_scores contains '{label}' not listed in mention_types"
                )
            if score is not None and not (0.0 <= float(score) <= 1.0):
                raise ValueError(f"confidence_scores['{label}'] must be 0.0-1.0")

        missing = [label for label in types if scores.get(label) is None]
        if missing:
            raise ValueError(
                f"confidence_scores missing required labels: {', '.join(sorted(missing))}"
            )
        return self


# =============================================================================
# Adoption Type Classifier
# =============================================================================


class AdoptionType(str, Enum):
    """Types of AI adoption."""

    non_llm = "non_llm"
    llm = "llm"
    agentic = "agentic"


class AdoptionConfidenceScores(BaseModel):
    """Confidence scores for each adoption type."""

    non_llm: Optional[float] = Field(
        default=None,
        description="Traditional ML/AI confidence (0.0-1.0): forecasting, analytics, CV, anomaly detection",
    )
    llm: Optional[float] = Field(
        default=None,
        description="LLM confidence (0.0-1.0): GPT, ChatGPT, Gemini, Claude, copilots",
    )
    agentic: Optional[float] = Field(
        default=None,
        description="Agentic AI confidence (0.0-1.0): autonomous agents, limited human intervention",
    )


class AdoptionTypeResponse(BaseModel):
    """Response schema for adoption type classification."""

    adoption_confidences: AdoptionConfidenceScores = Field(
        description=(
            "Confidence scores (0.0-1.0) for each adoption type. "
            "0.0-0.2=unclear/generic, 0.3-0.6=implied/vague, 0.7-1.0=explicit."
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale.",
    )


# =============================================================================
# Risk Classifier
# =============================================================================


class RiskType(str, Enum):
    """AI risk categories taxonomy."""

    strategic_market = "strategic_market"
    operational_technical = "operational_technical"
    cybersecurity = "cybersecurity"
    workforce_impacts = "workforce_impacts"
    regulatory_compliance = "regulatory_compliance"
    information_integrity = "information_integrity"
    reputational_ethical = "reputational_ethical"
    third_party_supply_chain = "third_party_supply_chain"
    environmental_impact = "environmental_impact"
    national_security = "national_security"
    none = "none"


class RiskConfidenceScores(BaseModel):
    """Confidence scores for each risk type."""

    strategic_market: Optional[float] = Field(
        default=None,
        description="Strategic/market risk confidence (0.0-1.0): failure to adopt, competitive disadvantage",
    )
    operational_technical: Optional[float] = Field(
        default=None,
        description="Operational/technical risk confidence (0.0-1.0): model failures, bias, reliability",
    )
    cybersecurity: Optional[float] = Field(
        default=None,
        description="Cybersecurity risk confidence (0.0-1.0): AI-enabled attacks, vulnerabilities",
    )
    workforce_impacts: Optional[float] = Field(
        default=None,
        description="Workforce impact confidence (0.0-1.0): job displacement, skills obsolescence",
    )
    regulatory_compliance: Optional[float] = Field(
        default=None,
        description="Regulatory/compliance risk confidence (0.0-1.0): AI Act, GDPR, legal liability",
    )
    information_integrity: Optional[float] = Field(
        default=None,
        description="Information integrity confidence (0.0-1.0): misinformation, deepfakes",
    )
    reputational_ethical: Optional[float] = Field(
        default=None,
        description="Reputational/ethical risk confidence (0.0-1.0): public trust, bias concerns",
    )
    third_party_supply_chain: Optional[float] = Field(
        default=None,
        description="Third-party/supply chain risk confidence (0.0-1.0): vendor reliance",
    )
    environmental_impact: Optional[float] = Field(
        default=None,
        description="Environmental impact confidence (0.0-1.0): energy consumption, carbon",
    )
    national_security: Optional[float] = Field(
        default=None,
        description="National security risk confidence (0.0-1.0): geopolitical, export controls",
    )
    none: Optional[float] = Field(
        default=None,
        description="Too vague to assign (0.0-1.0)",
    )


class RiskResponse(BaseModel):
    """Response schema for risk classification."""

    risk_types: List[RiskType] = Field(
        description=(
            "Detected AI risk categories (non-mutually exclusive). "
            "Use 'none' if the excerpt is too vague to assign specific categories."
        )
    )
    confidence_scores: RiskConfidenceScores = Field(
        description=(
            "Confidence per detected risk type (0.0-1.0). "
            "0.0=no evidence, 0.2=implied, 0.5=plausible, 0.8=explicit, 0.95=unambiguous."
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale.",
    )


# =============================================================================
# Vendor Classifier
# =============================================================================


class VendorTag(str, Enum):
    """Known AI vendor tags."""

    google = "google"
    microsoft = "microsoft"
    openai = "openai"
    anthropic = "anthropic"
    meta = "meta"
    internal = "internal"
    undisclosed = "undisclosed"
    other = "other"


class VendorConfidenceScores(BaseModel):
    """Confidence scores for each vendor tag."""

    google: Optional[float] = Field(
        default=None,
        description="Google/Vertex AI/Gemini confidence (0.0-1.0)",
    )
    microsoft: Optional[float] = Field(
        default=None,
        description="Microsoft/Azure/Copilot confidence (0.0-1.0)",
    )
    openai: Optional[float] = Field(
        default=None,
        description="OpenAI/GPT/ChatGPT confidence (0.0-1.0)",
    )
    anthropic: Optional[float] = Field(
        default=None,
        description="Anthropic/Claude confidence (0.0-1.0)",
    )
    meta: Optional[float] = Field(
        default=None,
        description="Meta/Llama confidence (0.0-1.0)",
    )
    internal: Optional[float] = Field(
        default=None,
        description="In-house/proprietary AI confidence (0.0-1.0)",
    )
    undisclosed: Optional[float] = Field(
        default=None,
        description="Third-party mentioned but not named confidence (0.0-1.0)",
    )
    other: Optional[float] = Field(
        default=None,
        description="Named vendor not in list confidence (0.0-1.0)",
    )


class VendorResponse(BaseModel):
    """Response schema for vendor extraction."""

    vendor_confidences: VendorConfidenceScores = Field(
        description="Confidence scores (0.0-1.0) for each vendor tag."
    )
    other_vendor: Optional[str] = Field(
        default=None,
        description="Name of vendor if 'other' has non-zero confidence.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale.",
    )


# =============================================================================
# Substantiveness Classifier
# =============================================================================


class SubstantivenessLevel(str, Enum):
    """Levels of substantiveness in AI disclosures."""

    boilerplate = "boilerplate"
    contextual = "contextual"
    substantive = "substantive"


class SubstantivenessScores(BaseModel):
    """Confidence scores for each substantiveness level."""

    boilerplate: Optional[float] = Field(
        default=None,
        description="Generic legal phrasing confidence (0.0-1.0)",
    )
    contextual: Optional[float] = Field(
        default=None,
        description="Sector-relevant but non-specific confidence (0.0-1.0)",
    )
    substantive: Optional[float] = Field(
        default=None,
        description="Named systems, quantified impact confidence (0.0-1.0)",
    )


class SubstantivenessResponse(BaseModel):
    """Response schema for substantiveness classification."""

    substantiveness_scores: SubstantivenessScores = Field(
        description=(
            "Confidence scores (0.0-1.0) for each substantiveness level. "
            "0.0=no evidence, 0.2=weak, 0.5=plausible, 0.8=clear, 0.95=unambiguous."
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale.",
    )


# =============================================================================
# Harms Classifier
# =============================================================================


class HarmsResponse(BaseModel):
    """Response schema for harms classification."""

    harms_mentioned: bool = Field(
        description="Whether AI-related harms are mentioned in the excerpt."
    )
    confidence: float = Field(
        description=(
            "Confidence in the classification (0.0-1.0). "
            "0.0=no evidence, 0.2=implied, 0.5=plausible, 0.8=explicit, 0.95=unambiguous."
        )
    )
    evidence: Optional[List[str]] = Field(
        default=None,
        description="Up to 5 key quotes supporting the classification.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale.",
    )
