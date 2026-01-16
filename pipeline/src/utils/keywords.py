"""Keyword patterns used for AI mention detection."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class KeywordPattern:
    """Keyword pattern definition."""

    name: str
    pattern: str


AI_KEYWORD_PATTERNS: List[KeywordPattern] = [
    KeywordPattern("artificial_intelligence", r"\bartificial\s+intelligence\b"),
    KeywordPattern("ai", r"\bai\b"),
    KeywordPattern("machine_learning", r"\bmachine\s+learning\b"),
    KeywordPattern("ml", r"\bml\b"),
    KeywordPattern("deep_learning", r"\bdeep\s+learning\b"),
    KeywordPattern("neural_network", r"\bneural\s+network"),
    KeywordPattern("large_language_model", r"\blarge\s+language\s+model"),
    KeywordPattern("llm", r"\bllm\b"),
    KeywordPattern("generative_ai", r"\bgenerative\s+(?:ai|model)"),
    KeywordPattern("gen_ai", r"\bgen\s+ai\b"),
    KeywordPattern("transformer", r"\btransformer"),
    KeywordPattern("natural_language_processing", r"\bnatural\s+language\s+processing\b"),
    KeywordPattern("nlp", r"\bnlp\b"),
    KeywordPattern("computer_vision", r"\bcomputer\s+vision\b"),
    KeywordPattern("image_recognition", r"\bimage\s+recognition\b"),
    KeywordPattern("intelligent_automation", r"\bintelligent\s+automation\b"),
    KeywordPattern("rpa", r"\brobotic\s+process\s+automation\b"),
    KeywordPattern("rpa_acronym", r"\brpa\b"),
    KeywordPattern("predictive_analytics", r"\bpredictive\s+analytics?\b"),
    KeywordPattern("data_analytics", r"\bdata\s+analytics?\b"),
    KeywordPattern("chatbot", r"\bchatbot"),
    KeywordPattern("virtual_assistant", r"\bvirtual\s+assistant"),
    KeywordPattern("recommendation_system", r"\brecommendation\s+(?:engine|system|algorithm)"),
    KeywordPattern("autonomous", r"\bautonomous"),
    KeywordPattern("ai_compound", r"\bai[-](?:powered|driven|enabled|based)"),
    KeywordPattern("algorithmic", r"\balgorithm(?:ic)?\s+(?:trading|decision|bias)"),
]
