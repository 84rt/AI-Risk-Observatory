"""LLM-based classification for AI risk mentions using Google Gemini."""

import json
import logging
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .chunker import CandidateSpan

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Classification:
    """LLM classification result for a text span."""

    # Relevance
    is_relevant: bool

    # If not relevant, remaining fields are None
    mention_type: Optional[str] = None
    ai_specificity: Optional[str] = None
    frontier_tech_flag: Optional[bool] = None

    # Risk classification
    tier_1_category: Optional[str] = None
    tier_2_driver: Optional[str] = None

    # Specificity and materiality
    specificity_level: Optional[str] = None
    materiality_signal: Optional[str] = None

    # Governance
    mitigation_mentioned: Optional[bool] = None
    governance_maturity: Optional[str] = None

    # Metadata
    confidence_score: Optional[float] = None
    reasoning_summary: Optional[str] = None


class LLMClassifier:
    """Classifier using Google Gemini for AI risk detection."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the classifier.

        Args:
            api_key: Gemini API key. If not provided, uses settings.
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = settings.gemini_model

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize model with JSON response mode
        generation_config = {
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_tokens,
            "response_mime_type": "application/json",
        }

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=settings.retry_delay, max=30)
    )
    def classify_span(
        self,
        candidate: CandidateSpan
    ) -> Classification:
        """Classify a candidate span for AI relevance and risk.

        Args:
            candidate: CandidateSpan to classify

        Returns:
            Classification result
        """
        logger.debug(f"Classifying span {candidate.span_id}")

        # Build the prompt
        prompt = self._build_classification_prompt(candidate)

        # Call Gemini
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
        except Exception as e:
            logger.error(f"Gemini API error for {candidate.span_id}: {e}")
            # Return not relevant if API fails
            return Classification(is_relevant=False)

        # Parse response
        try:
            classification_data = self._parse_response(response_text)
            classification = Classification(**classification_data)
            logger.debug(
                f"Span {candidate.span_id}: "
                f"relevant={classification.is_relevant}"
            )
            return classification
        except Exception as e:
            logger.error(
                f"Failed to parse classification response for {candidate.span_id}: {e}"
            )
            logger.error(f"Response was: {response_text}")
            # Return not relevant if parsing fails
            return Classification(is_relevant=False)

    def _build_classification_prompt(self, candidate: CandidateSpan) -> str:
        """Build the classification prompt.

        Args:
            candidate: CandidateSpan to classify

        Returns:
            Prompt string
        """
        prompt = f"""You are an expert analyst for the UK AI Safety Institute, analyzing company annual reports for mentions of AI-related risks and adoption.

## INPUT
Company: {candidate.firm_name}
Sector: {candidate.sector}
Report Year: {candidate.report_year}
Report Section: {candidate.report_section or "Unknown"}

Excerpt:
\"\"\"
{candidate.text}
\"\"\"

## TASK
Analyze this excerpt and provide a structured classification.

### Step 1: Is this AI-relevant?
Is this excerpt about AI, machine learning, algorithms, or automated decision-making in a context of risk, adoption, governance, or incidents?

Answer: [Yes/No]
If No, stop here.

### Step 2: Mention Type
What is this excerpt primarily about?
- risk_statement: Explicit discussion of AI-related risk or downside
- adoption_use_case: Description of AI deployment or planned use
- governance_mitigation: Controls, policies, or risk management for AI
- incident_event: Concrete failure, breach, or regulatory action
- regulatory_environment: Discussion of AI regulation or compliance
- strategy_opportunity: Strategic AI discussion with implicit risk context

Answer: [mention_type]

### Step 3: AI Specificity
- ai_specific: Explicit AI/ML terms (AI, machine learning, LLM, neural network, etc.)
- automation_general: Generic automation without clear AI reference

Answer: [ai_specificity]

### Step 4: Frontier Technology
Does this mention frontier AI technologies (large language models, generative AI, foundation models, GPT, Claude, Gemini, etc.)?

Answer: [true/false]

### Step 5: Risk Classification (if risk-related)
Tier 1 Category (select one, or null if not risk-related):
- operational_reliability: System failures, model errors, outages
- security_malicious_use: Cyber attacks, deepfakes, fraud
- legal_regulatory_compliance: AI Act, GDPR, IP, liability
- workforce_human_capital: Job displacement, skills, talent
- societal_ethical_reputational: Bias, misinformation, trust
- frontier_systemic: Loss of control, systemic risk

Answer: [tier_1_category or null]

Tier 2 Driver (select one if applicable, or null):
- third_party_dependence (parent: operational_reliability)
- hallucination_accuracy (parent: operational_reliability)
- model_drift_degradation (parent: operational_reliability)
- cyber_enablement (parent: security_malicious_use)
- adversarial_attacks (parent: security_malicious_use)
- deepfakes_synthetic_media (parent: security_malicious_use)
- data_privacy_leakage (parent: legal_regulatory_compliance)
- ip_copyright (parent: legal_regulatory_compliance)
- regulatory_uncertainty (parent: legal_regulatory_compliance)
- job_displacement (parent: workforce_human_capital)
- skill_obsolescence (parent: workforce_human_capital)
- shadow_ai (parent: workforce_human_capital)
- bias_discrimination (parent: societal_ethical_reputational)
- misinformation_content (parent: societal_ethical_reputational)
- trust_reputation (parent: societal_ethical_reputational)
- concentration_risk (parent: frontier_systemic)
- loss_of_control (parent: frontier_systemic)

Answer: [tier_2_driver or null]

### Step 6: Specificity Level
- boilerplate: Generic language applicable to any firm
- contextual: Sector/company-relevant but not specific
- concrete: Named systems, quantified, or detailed

Answer: [specificity_level]

### Step 7: Materiality Signal
Based on language cues ("material", "significant", "principal", etc.):
- low / medium / high / unspecified

Answer: [materiality_signal]

### Step 8: Governance & Mitigation
Is mitigation or governance discussed?
Answer: [true/false]

If yes, what level of governance maturity is indicated?
- none / basic / intermediate / advanced

Answer: [governance_maturity]

### Step 9: Confidence
How confident are you in this classification? (0.0 - 1.0)
- 0.9-1.0: Very clear, unambiguous
- 0.7-0.89: Clear, minor ambiguity
- 0.5-0.69: Moderate uncertainty
- Below 0.5: High uncertainty, needs human review

Answer: [confidence_score]

### Step 10: Reasoning
Provide a 1-2 sentence explanation of your classification.

Answer: [reasoning_summary]

## OUTPUT FORMAT
Return your response as JSON ONLY (no other text).

If the excerpt IS AI-relevant, return:
{{
  "is_relevant": true,
  "mention_type": "risk_statement",
  "ai_specificity": "ai_specific",
  "frontier_tech_flag": false,
  "tier_1_category": "operational_reliability",
  "tier_2_driver": "hallucination_accuracy",
  "specificity_level": "contextual",
  "materiality_signal": "medium",
  "mitigation_mentioned": true,
  "governance_maturity": "basic",
  "confidence_score": 0.85,
  "reasoning_summary": "Clear mention of AI model accuracy risks in customer-facing context, with generic policy reference but no specific controls."
}}

If the excerpt is NOT AI-relevant, return:
{{
  "is_relevant": false
}}
"""
        return prompt

    @staticmethod
    def _parse_response(response_text: str) -> dict:
        """Parse the JSON response from Gemini.

        Args:
            response_text: Response text from Gemini

        Returns:
            Dict with classification data
        """
        # Gemini with JSON mode should return clean JSON
        # But handle markdown code blocks just in case
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text.strip()

        # Parse JSON
        data = json.loads(json_text)

        # Validate is_relevant is present
        if "is_relevant" not in data:
            raise ValueError("Response missing 'is_relevant' field")

        return data


def classify_candidates(
    candidates: list[CandidateSpan],
    batch_size: int = 10
) -> list[tuple[CandidateSpan, Classification]]:
    """Classify a batch of candidate spans.

    Args:
        candidates: List of CandidateSpan objects
        batch_size: Number to process before logging progress

    Returns:
        List of (candidate, classification) tuples
    """
    classifier = LLMClassifier()
    results = []

    total = len(candidates)
    for i, candidate in enumerate(candidates, 1):
        if i % batch_size == 0:
            logger.info(f"Classified {i}/{total} spans...")

        classification = classifier.classify_span(candidate)
        results.append((candidate, classification))

    logger.info(f"Completed classification of {total} spans")

    return results
