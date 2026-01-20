"""Keyword patterns used for AI mention detection."""

from dataclasses import dataclass
import re
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class KeywordPattern:
    """Keyword pattern definition."""

    name: str
    pattern: str
    case_sensitive: bool = False


def compile_keyword_patterns(
    patterns: Optional[List[KeywordPattern]] = None,
) -> List[Tuple[str, re.Pattern]]:
    """Compile keyword patterns with per-pattern case sensitivity."""
    compiled: List[Tuple[str, re.Pattern]] = []
    for kp in patterns or AI_KEYWORD_PATTERNS:
        flags = 0 if kp.case_sensitive else re.IGNORECASE
        compiled.append((kp.name, re.compile(kp.pattern, flags)))
    return compiled


AI_KEYWORD_PATTERNS: List[KeywordPattern] = [
    KeywordPattern("artificial_intelligence", r"\bartificial\s+intelligence\b"),
    KeywordPattern(
        "ai",
        r"\bAI\b",
        case_sensitive=True,
    ),  # Uppercase only to avoid "Shanghai", "Chairman", etc.
    KeywordPattern("a_i", r"\ba\.i\.\b"),
    KeywordPattern("machine_learning", r"\bmachine\s+learning\b"),
    KeywordPattern("ml", r"\bML\b", case_sensitive=True),
    KeywordPattern("machine_learned", r"\bmachine[-\s]+learned\b"),
    KeywordPattern("deep_learning", r"\bdeep\s+learning\b"),
    KeywordPattern("neural_network", r"\bneural\s+network"),
    KeywordPattern("neural_net", r"\bneural\s+net(?:s)?\b"),
    KeywordPattern("deep_neural_network", r"\bdeep\s+neural\s+network(?:s)?\b"),
    KeywordPattern("large_language_model", r"\blarge\s+language\s+model"),
    KeywordPattern("llm", r"\bllm\b"),
    KeywordPattern("language_model", r"\blanguage\s+model(?:s)?\b"),
    KeywordPattern("foundation_model", r"\bfoundation\s+model(?:s)?\b"),
    KeywordPattern("generative_ai", r"\bgenerative\s+(?:ai|model)"),
    KeywordPattern("gen_ai", r"\bgen\s+ai\b"),
    KeywordPattern("natural_language_processing", r"\bnatural\s+language\s+processing\b"),
    KeywordPattern("nlp", r"\bnlp\b"),
    KeywordPattern("computer_vision", r"\bcomputer\s+vision\b"),
    KeywordPattern("image_recognition", r"\bimage\s+recognition\b"),
    KeywordPattern("image_classification", r"\bimage\s+classification\b"),
    KeywordPattern("object_detection", r"\bobject\s+detection\b"),
    KeywordPattern("semantic_segmentation", r"\bsemantic\s+segmentation\b"),
    KeywordPattern("intelligent_automation", r"\bintelligent\s+automation\b"),
    KeywordPattern("rpa", r"\brobotic\s+process\s+automation\b"),
    KeywordPattern("rpa_acronym", r"\brpa\b"),
    KeywordPattern("predictive_analytics", r"\bpredictive\s+analytics?\b"),
    KeywordPattern("chatbot", r"\bchatbot"),
    KeywordPattern("virtual_assistant", r"\bvirtual\s+assistant"),
    KeywordPattern("recommendation_system", r"\brecommendation\s+(?:engine|system|algorithm)"),
    KeywordPattern("autonomous", r"\bautonomous\b"),
    KeywordPattern("algorithmic", r"\balgorithm(?:ic)?\s+(?:trading|decision|bias)"),
    KeywordPattern("chatgpt", r"\bchatgpt\b"),
    KeywordPattern("gpt", r"\bgpt-?\d+\b"),
    KeywordPattern("gpt_family", r"\bgpt\b"),
    KeywordPattern("claude", r"\bclaude\b"),
    KeywordPattern("llama", r"\bllama\b"),
    KeywordPattern("stable_diffusion", r"\bstable\s+diffusion\b"),
    KeywordPattern("speech_recognition", r"\bspeech\s+recognition\b"),
    KeywordPattern("speech_to_text", r"\bspeech[-\s]+to[-\s]+text\b"),
    KeywordPattern("text_to_speech", r"\btext[-\s]+to[-\s]+speech\b"),
    KeywordPattern("text_to_image", r"\btext[-\s]+to[-\s]+image\b"),
    KeywordPattern("text_to_video", r"\btext[-\s]+to[-\s]+video\b"),
    KeywordPattern("image_generation", r"\bimage\s+generation\b"),
    KeywordPattern("copilot", r"\bcopilot\b"),
    KeywordPattern("openai", r"\bopenai\b"),
    KeywordPattern("anthropic", r"\banthropic\b"),
    KeywordPattern("deepmind", r"\bdeepmind\b"),
    # AI governance and risk terms
    KeywordPattern("explainable_ai", r"\bexplainable\s+ai\b"),
    KeywordPattern("xai", r"\bxai\b"),
    KeywordPattern("responsible_ai", r"\bresponsible\s+ai\b"),
    KeywordPattern("ai_ethics", r"\bai\s+ethics\b"),
    KeywordPattern("ethical_ai", r"\bethical\s+ai\b"),
    KeywordPattern("ai_governance", r"\bai\s+governance\b"),
    KeywordPattern("ai_safety", r"\bai\s+safety\b"),
    KeywordPattern("ai_risk", r"\bai\s+risk(?:s)?\b"),
    KeywordPattern("ai_audit", r"\bai\s+audit(?:s|ing)?\b"),
    KeywordPattern("ai_regulation", r"\bai\s+regulation(?:s)?\b"),
    KeywordPattern("ai_compliance", r"\bai\s+compliance\b"),
    # Additional technical terms
    KeywordPattern("frontier_model", r"\bfrontier\s+model(?:s)?\b"),
    KeywordPattern("reinforcement_learning", r"\breinforcement\s+learning\b"),
    KeywordPattern("agentic", r"\bagentic\b"),
    KeywordPattern("ai_agent", r"\bai\s+agent(?:s)?\b"),
    KeywordPattern("intelligent_agent", r"\bintelligent\s+agent(?:s)?\b"),
    KeywordPattern("multimodal", r"\bmultimodal\b"),
    KeywordPattern("cognitive_computing", r"\bcognitive\s+computing\b"),
]
