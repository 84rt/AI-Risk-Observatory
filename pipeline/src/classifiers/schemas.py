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

    chunk_id: Optional[str] = Field(
        default=None,
        description="Echoed chunk identifier when provided in the prompt.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale.",
    )
    mention_types: List[MentionType] = Field(
        description=(
            "Detected mention types. "
            "'adoption': Real deployment/implementation/pilot of AI (not intent/strategy alone). "
            "'risk': AI directly attributed as risk source. "
            "'harm': Past harms caused by AI. "
            "'vendor': Named AI vendor (Microsoft, Google, OpenAI, AWS, etc.). "
            "'general_ambiguous': Vague AI mentions not fitting other categories; try use alone. "
            "'none': No explicit AI/ML/LLM mention or false positive."
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
# Substantiveness (shared enum, used by multiple response models)
# =============================================================================


class SubstantivenessLevel(str, Enum):
    """How tangible/concrete an AI disclosure is.

    boilerplate:  Pure jargon, no information content. Could appear in any report unchanged.
    moderate:     Identifies a specific area or application but lacks concrete details.
    substantive:  Contains specifics: named systems, quantified impact, or tangible commitments.
    """

    boilerplate = "boilerplate"
    moderate = "moderate"
    substantive = "substantive"


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
    substantiveness: SubstantivenessLevel = Field(
        description=(
            "How tangible is the AI adoption disclosure? "
            "'boilerplate': generic phrasing with no information content (e.g., 'We leverage AI to drive innovation'). "
            "'moderate': identifies a specific use case or domain but lacks detail (e.g., 'We use AI in our underwriting process'). "
            "'substantive': names systems, quantifies impact, or explains what/how/why (e.g., 'We deployed GPT-4 for document review, cutting processing time by 40%')."
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

    strategic_competitive = "strategic_competitive"
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

    strategic_competitive: Optional[float] = Field(
        default=None,
        description="Strategic/competitive risk confidence (0.0-1.0): failure to adopt, competitive disadvantage",
    )
    operational_technical: Optional[float] = Field(
        default=None,
        description="Operational/technical risk confidence (0.0-1.0): model failures, reliability",
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
        description="Regulatory/compliance risk confidence (0.0-1.0): AI Act, privacy, IP/copyright, legal liability",
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
    substantiveness: SubstantivenessLevel = Field(
        description=(
            "How tangible is the AI risk disclosure? "
            "'boilerplate': generic risk language with no information content (e.g., 'AI poses risks to our business'). "
            "'moderate': identifies a specific risk area but no mitigation or detail (e.g., 'AI regulation may affect our compliance obligations'). "
            "'substantive': describes specific risk mechanisms and tangible mitigation actions or commitments (e.g., 'We allocated EUR 5M to reclassify 3 high-risk AI systems under the EU AI Act by 2025')."
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

    amazon = "amazon"
    google = "google"
    microsoft = "microsoft"
    openai = "openai"
    anthropic = "anthropic"
    meta = "meta"
    internal = "internal"
    undisclosed = "undisclosed"
    other = "other"


class VendorSignalScores(BaseModel):
    """Signal strengths for detected vendor tags. Only set scores for tags listed in vendor_tags."""

    amazon: Optional[int] = Field(
        default=None,
        description="Amazon/AWS/Bedrock/CodeWhisperer signal (1-3)",
    )
    google: Optional[int] = Field(
        default=None,
        description="Google/Vertex AI/Gemini signal (1-3)",
    )
    microsoft: Optional[int] = Field(
        default=None,
        description="Microsoft/Azure/Copilot signal (1-3)",
    )
    openai: Optional[int] = Field(
        default=None,
        description="OpenAI/GPT/ChatGPT signal (1-3)",
    )
    anthropic: Optional[int] = Field(
        default=None,
        description="Anthropic/Claude signal (1-3)",
    )
    meta: Optional[int] = Field(
        default=None,
        description="Meta/Llama signal (1-3)",
    )
    internal: Optional[int] = Field(
        default=None,
        description="In-house/proprietary AI signal (1-3)",
    )
    undisclosed: Optional[int] = Field(
        default=None,
        description="Third-party mentioned but not named signal (1-3)",
    )
    other: Optional[int] = Field(
        default=None,
        description="Named vendor not in list signal (1-3)",
    )


class VendorResponse(BaseModel):
    """Response schema for vendor extraction."""

    chunk_id: Optional[str] = Field(
        default=None,
        description="Echoed chunk identifier when provided in the prompt.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale.",
    )
    vendor_tags: List[VendorTag] = Field(
        description=(
            "Detected vendor tags. "
            "'amazon': Amazon/AWS/Bedrock/CodeWhisperer. "
            "'google': Google/Vertex AI/Gemini/DeepMind. "
            "'microsoft': Azure AI/Copilot/Azure OpenAI. "
            "'openai': OpenAI/GPT/ChatGPT. "
            "'anthropic': Anthropic/Claude. "
            "'meta': Meta AI/Llama. "
            "'internal': In-house/proprietary AI. "
            "'undisclosed': Third-party vendor not named. "
            "'other': Named vendor not in list."
        )
    )
    vendor_signals: VendorSignalScores = Field(
        description=(
            "Signal strength per detected vendor (1=weak implicit, 2=strong implicit, 3=explicit). "
            "Only include signals for tags listed in vendor_tags."
        )
    )
    other_vendor: Optional[str] = Field(
        default=None,
        description="Name of vendor if 'other' is in vendor_tags.",
    )

    @model_validator(mode="after")
    def validate_signals(self) -> "VendorResponse":
        tags = {vt.value for vt in self.vendor_tags}

        signals = self.vendor_signals.model_dump()
        for label, signal in signals.items():
            if signal is not None and label not in tags:
                raise ValueError(
                    f"vendor_signals contains '{label}' not listed in vendor_tags"
                )
            if signal is not None and signal not in (1, 2, 3):
                raise ValueError(f"vendor_signals['{label}'] must be 1, 2, or 3")

        missing = [label for label in tags if signals.get(label) is None]
        if missing:
            raise ValueError(
                f"vendor_signals missing required tags: {', '.join(sorted(missing))}"
            )
        return self


# =============================================================================
# Substantiveness Classifier
# =============================================================================


class SubstantivenessScores(BaseModel):
    """Confidence scores for each substantiveness level."""

    boilerplate: Optional[float] = Field(
        default=None,
        description="Generic jargon, no information content (0.0-1.0)",
    )
    moderate: Optional[float] = Field(
        default=None,
        description="Identifies area/application but lacks concrete detail (0.0-1.0)",
    )
    substantive: Optional[float] = Field(
        default=None,
        description="Named systems, quantified impact, tangible commitments (0.0-1.0)",
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
