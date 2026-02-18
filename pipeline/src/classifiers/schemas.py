"""Pydantic response schemas for LLM classifiers.

These schemas serve dual purposes:
1. Type validation for LLM responses
2. Field descriptions provide context to Gemini's response_schema

Note: Dict types are avoided for Gemini compatibility. Instead, we use
nested objects with explicit properties for known keys.
"""

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, conint, model_validator


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


class AdoptionSignalEntry(BaseModel):
    """Single adoption signal entry."""

    type: AdoptionType = Field(
        description="Adoption type label."
    )
    signal: conint(ge=0, le=3) = Field(
        description="Signal strength (0-3): 0=absent, 1=weak, 2=strong implicit, 3=explicit",
    )


class AdoptionTypeResponse(BaseModel):
    """Response schema for adoption type classification."""

    chunk_id: Optional[str] = Field(
        default=None,
        description="Echoed chunk identifier when provided in the prompt.",
    )
    adoption_signals: List[AdoptionSignalEntry] = Field(
        description=(
            "List of adoption signals (one entry per adoption type). "
            "Each entry has {type, signal} with signal in {0,1,2,3}."
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

    @model_validator(mode="after")
    def validate_adoption_signals(self) -> "AdoptionTypeResponse":
        types = [entry.type for entry in self.adoption_signals]
        if len(types) != len(set(types)):
            raise ValueError("adoption_signals must include each adoption type at most once")
        missing = set(AdoptionType) - set(types)
        if missing:
            raise ValueError(
                f"adoption_signals missing required types: {', '.join(sorted([m.value for m in missing]))}"
            )
        return self


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

    strategic_competitive: conint(ge=0, le=3) = Field(
        description="Strategic/competitive risk signal (0-3): failure to adopt, competitive disadvantage",
    )
    operational_technical: conint(ge=0, le=3) = Field(
        description="Operational/technical risk signal (0-3): model failures, reliability",
    )
    cybersecurity: conint(ge=0, le=3) = Field(
        description="Cybersecurity risk signal (0-3): AI-enabled attacks, vulnerabilities",
    )
    workforce_impacts: conint(ge=0, le=3) = Field(
        description="Workforce impact signal (0-3): job displacement, skills obsolescence",
    )
    regulatory_compliance: conint(ge=0, le=3) = Field(
        description="Regulatory/compliance risk signal (0-3): AI Act, privacy, IP/copyright, legal liability",
    )
    information_integrity: conint(ge=0, le=3) = Field(
        description="Information integrity signal (0-3): misinformation, deepfakes",
    )
    reputational_ethical: conint(ge=0, le=3) = Field(
        description="Reputational/ethical risk signal (0-3): public trust, bias concerns",
    )
    third_party_supply_chain: conint(ge=0, le=3) = Field(
        description="Third-party/supply chain risk signal (0-3): vendor reliance",
    )
    environmental_impact: conint(ge=0, le=3) = Field(
        description="Environmental impact signal (0-3): energy consumption, carbon",
    )
    national_security: conint(ge=0, le=3) = Field(
        description="National security risk signal (0-3): geopolitical, export controls",
    )
    none: conint(ge=0, le=3) = Field(
        description="Too vague to assign (0-3)",
    )


class RiskSignalEntry(BaseModel):
    """Single risk signal entry for an applied risk type."""

    type: RiskType = Field(
        description="Risk type label."
    )
    signal: conint(ge=1, le=3) = Field(
        description="Signal strength (1-3): 1=weak implicit, 2=strong implicit, 3=explicit",
    )


class RiskResponse(BaseModel):
    """Response schema for risk classification."""

    chunk_id: Optional[str] = Field(
        default=None,
        description="Echoed chunk identifier when provided in the prompt.",
    )
    risk_types: List[RiskType] = Field(
        description=(
            "Detected AI risk categories (non-mutually exclusive). "
            "Use 'none' if the excerpt is too vague to assign specific categories."
        )
    )
    risk_signals: List[RiskSignalEntry] = Field(
        description=(
            "Signal scores (1-3) for risk types listed in risk_types only. "
            "1=weak implicit, 2=strong implicit, 3=explicit."
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

    @model_validator(mode="after")
    def validate_risk_signals(self) -> "RiskResponse":
        types = [rt.value for rt in self.risk_types]
        if not types:
            raise ValueError("risk_types must not be empty")
        if "none" in types and len(types) > 1:
            raise ValueError("risk_types cannot include 'none' with other labels")

        seen = set()
        for entry in self.risk_signals:
            key = entry.type.value
            if key in seen:
                raise ValueError(f"risk_signals contains duplicate type: {key}")
            seen.add(key)

        type_set = set(types)
        if seen != type_set:
            missing = sorted(type_set - seen)
            extra = sorted(seen - type_set)
            details = []
            if missing:
                details.append(f"missing types: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected types: {', '.join(extra)}")
            raise ValueError(f"risk_signals must match risk_types exactly ({'; '.join(details)})")

        return self


# =============================================================================
# Open Risk Discovery Classifier
# =============================================================================


class OpenRiskSignalEntry(BaseModel):
    """Single open-taxonomy risk signal entry."""

    type: str = Field(
        description=(
            "Emergent risk label in snake_case, or 'none' when no attributable AI-risk exists."
        )
    )
    signal: conint(ge=1, le=3) = Field(
        description="Signal strength (1-3): 1=weak implicit, 2=strong implicit, 3=explicit",
    )


class OpenRiskLabelDefinitionEntry(BaseModel):
    """One-line definition for an emergent risk label."""

    type: str = Field(
        description="Emergent risk label in snake_case."
    )
    definition: str = Field(
        min_length=8,
        max_length=240,
        description="One-line definition grounded in the excerpt.",
    )


class OpenRiskEvidenceEntry(BaseModel):
    """Evidence snippet supporting an emergent risk label."""

    type: str = Field(
        description="Emergent risk label in snake_case."
    )
    snippet: str = Field(
        min_length=8,
        max_length=360,
        description="Quoted or near-quoted supporting snippet from the excerpt.",
    )


class OpenRiskResponse(BaseModel):
    """Response schema for conservative open-taxonomy AI risk discovery."""

    chunk_id: Optional[str] = Field(
        default=None,
        description="Echoed chunk identifier when provided in the prompt.",
    )
    risk_types: List[str] = Field(
        description=(
            "Detected emergent AI risk labels in snake_case. "
            "Use ['none'] when no attributable AI-risk category is supported."
        ),
        max_length=3,
    )
    risk_signals: List[OpenRiskSignalEntry] = Field(
        description=(
            "Signal scores (1-3) for labels in risk_types only; one entry per label."
        )
    )
    label_definitions: List[OpenRiskLabelDefinitionEntry] = Field(
        description=(
            "One-line definitions for non-none labels; exactly one definition per label."
        )
    )
    evidence: List[OpenRiskEvidenceEntry] = Field(
        description=(
            "Evidence snippets for non-none labels; include at least one snippet per label."
        )
    )
    substantiveness: SubstantivenessLevel = Field(
        description=(
            "How tangible is the AI risk disclosure? "
            "'boilerplate': generic risk language with no information content. "
            "'moderate': specific risk area with limited mechanism/mitigation detail. "
            "'substantive': specific mechanism plus tangible mitigation actions/commitments."
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale.",
    )

    @staticmethod
    def _normalize_label(raw: object) -> str:
        token = str(raw).strip().lower()
        if token == "none":
            return token
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", token):
            raise ValueError(
                f"Invalid open risk label '{raw}'. Labels must be snake_case (2-64 chars)."
            )
        return token

    @model_validator(mode="after")
    def validate_open_risk_payload(self) -> "OpenRiskResponse":
        if not self.risk_types:
            raise ValueError("risk_types must not be empty")

        normalized_types = [self._normalize_label(rt) for rt in self.risk_types]
        if len(normalized_types) != len(set(normalized_types)):
            raise ValueError("risk_types must not contain duplicate labels")
        if "none" in normalized_types and len(normalized_types) > 1:
            raise ValueError("risk_types cannot include 'none' with other labels")
        if len(normalized_types) > 3:
            raise ValueError("risk_types must contain at most 3 labels")
        self.risk_types = normalized_types

        signal_labels: list[str] = []
        seen_signal_labels: set[str] = set()
        for entry in self.risk_signals:
            lbl = self._normalize_label(entry.type)
            if lbl in seen_signal_labels:
                raise ValueError(f"risk_signals contains duplicate label: {lbl}")
            entry.type = lbl
            seen_signal_labels.add(lbl)
            signal_labels.append(lbl)

        type_set = set(normalized_types)
        if set(signal_labels) != type_set:
            missing = sorted(type_set - set(signal_labels))
            extra = sorted(set(signal_labels) - type_set)
            details = []
            if missing:
                details.append(f"missing labels: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected labels: {', '.join(extra)}")
            raise ValueError(f"risk_signals must match risk_types exactly ({'; '.join(details)})")

        if type_set == {"none"}:
            if self.label_definitions:
                raise ValueError("label_definitions must be empty when risk_types is ['none']")
            if self.evidence:
                raise ValueError("evidence must be empty when risk_types is ['none']")
            return self

        definition_labels: list[str] = []
        seen_definition_labels: set[str] = set()
        for entry in self.label_definitions:
            lbl = self._normalize_label(entry.type)
            if lbl == "none":
                raise ValueError("label_definitions cannot include 'none'")
            if lbl in seen_definition_labels:
                raise ValueError(f"label_definitions contains duplicate label: {lbl}")
            entry.type = lbl
            seen_definition_labels.add(lbl)
            definition_labels.append(lbl)

        if set(definition_labels) != type_set:
            missing = sorted(type_set - set(definition_labels))
            extra = sorted(set(definition_labels) - type_set)
            details = []
            if missing:
                details.append(f"missing labels: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected labels: {', '.join(extra)}")
            raise ValueError(
                f"label_definitions must match non-none risk_types exactly ({'; '.join(details)})"
            )

        evidence_labels: set[str] = set()
        for entry in self.evidence:
            lbl = self._normalize_label(entry.type)
            if lbl == "none":
                raise ValueError("evidence cannot include 'none'")
            if lbl not in type_set:
                raise ValueError(
                    f"evidence contains label not present in risk_types: {lbl}"
                )
            entry.type = lbl
            evidence_labels.add(lbl)

        missing_evidence = sorted(type_set - evidence_labels)
        if missing_evidence:
            raise ValueError(
                f"evidence must include at least one snippet per risk label (missing: {', '.join(missing_evidence)})"
            )

        return self


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


class VendorEntry(BaseModel):
    """A single vendor detection with its signal strength."""

    vendor: VendorTag = Field(description="Detected vendor tag.")
    signal: int = Field(description="Signal strength: 1=weak implicit, 2=strong implicit, 3=explicit.")


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
    vendors: List[VendorEntry] = Field(
        description=(
            "Detected vendors with signal strengths. "
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
    other_vendor: Optional[str] = Field(
        default=None,
        description="Name of vendor if 'other' is in vendors list.",
    )

    @model_validator(mode="after")
    def validate_vendors(self) -> "VendorResponse":
        seen = set()
        has_other = False
        for entry in self.vendors:
            tag = entry.vendor.value if hasattr(entry.vendor, "value") else str(entry.vendor)
            if tag in seen:
                raise ValueError(f"Duplicate vendor: '{tag}'")
            seen.add(tag)
            if entry.signal not in (1, 2, 3):
                raise ValueError(f"Signal for '{tag}' must be 1, 2, or 3")
            if tag == "other":
                has_other = True

        if has_other and not self.other_vendor:
            raise ValueError("other_vendor must be set when 'other' is in vendors")
        if not has_other and self.other_vendor:
            raise ValueError("other_vendor should only be set when 'other' is in vendors")
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


class SubstantivenessResponseV2(BaseModel):
    """Standalone substantiveness classifier response (v2).

    Assigns a single substantiveness label with confidence and reasoning.
    Designed to run independently of mention_type classification.
    """

    chunk_id: Optional[str] = Field(
        default=None,
        description="Echoed chunk identifier when provided in the prompt.",
    )
    substantiveness: SubstantivenessLevel = Field(
        description=(
            "How tangible is the AI disclosure? "
            "'boilerplate': generic jargon, interchangeable across companies. "
            "'moderate': names a domain/application but lacks specifics. "
            "'substantive': names systems/vendors/metrics/timelines with concrete detail."
        )
    )
    confidence: float = Field(
        description="Classification confidence (0.0-1.0).",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of classification rationale (1-2 sentences).",
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
