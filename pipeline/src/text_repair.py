"""Routing and repair utilities for noisy extracted text."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .ixbrl_extractor import (
    TextSpan,
    _basic_normalize_text,
    _calculate_quality_metrics,
    _get_word_dict,
    _merge_single_letter_fragments,
    _repair_fragmented_words,
    _repair_fragmented_words_with_symspell,
    _split_concatenated_with_wordninja,
)

logger = logging.getLogger(__name__)


class KenLMScorer:
    """Optional KenLM scorer for quality routing and validation."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.model_path = self._resolve_model_path(model_path)
        if self.model_path:
            try:
                import kenlm  # type: ignore
                self.model = kenlm.Model(str(self.model_path))
            except Exception as exc:
                logger.warning(f"KenLM not available: {exc}")
                self.model = None

    def _resolve_model_path(self, model_path: Optional[Path]) -> Optional[Path]:
        if model_path is not None:
            return model_path if model_path.exists() else None
        env_path = os.getenv("KENLM_MODEL_PATH")
        if env_path:
            path = Path(env_path).expanduser()
            return path if path.exists() else None
        default_path = Path(__file__).resolve().parents[2] / "data" / "reference" / "kenlm.arpa.bin"
        return default_path if default_path.exists() else None

    def perplexity(self, text: str) -> Optional[float]:
        if self.model is None:
            return None
        try:
            return float(self.model.perplexity(text))
        except Exception as exc:
            logger.warning(f"KenLM perplexity failed: {exc}")
            return None


class LLMReconstructor:
    """Optional LLM-based reconstruction hook."""

    def __init__(self):
        self.enabled = False

    def reconstruct(self, text: str) -> str:
        if not self.enabled:
            return text
        return text


class TextRepairService:
    """Route spans to repair strategies based on quality metrics."""

    def __init__(self, kenlm_model_path: Optional[Path] = None):
        self.scorer = KenLMScorer(kenlm_model_path)
        self.llm = LLMReconstructor()
        self.words_dict = _get_word_dict()
        self.safe_connectors = {
            "and", "the", "for", "our", "are", "has", "was",
            "with", "from", "been", "this", "that", "will",
        }
        self.risky_connectors = {"of", "to", "in", "on", "at", "as", "or", "by", "its"}

    def repair_spans(self, spans: Iterable[TextSpan]) -> List[TextSpan]:
        repaired = []
        for span in spans:
            if getattr(span, "repaired", False):
                repaired.append(span)
                continue
            raw_text = span.raw_text if span.raw_text is not None else span.text
            quality = self._quality(raw_text, span.quality)
            repaired_text, level = self.repair_text(raw_text, quality)
            span.text = repaired_text
            span.repaired = True
            span.quality = {**quality, "repair_level": float(level)}
            repaired.append(span)
        return repaired

    def repair_text(self, text: str, quality: Dict[str, float]) -> tuple[str, int]:
        level = self._route_level(quality)
        if level == 0:
            candidate = self._level0(text)
        elif level == 1:
            candidate = self._level1(text)
        elif level == 2:
            candidate = self._level2(text)
        else:
            candidate = self._level3(text)

        verified = self._verify(text, candidate)
        return verified, level

    def _quality(self, text: str, quality: Optional[Dict[str, float]]) -> Dict[str, float]:
        base = quality if quality else _calculate_quality_metrics(text)
        if "perplexity" not in base or base["perplexity"] == 0:
            perplexity = self.scorer.perplexity(text)
            if perplexity is not None:
                base = {**base, "perplexity": perplexity}
        return base

    def _route_level(self, quality: Dict[str, float]) -> int:
        perplexity = quality.get("perplexity")
        if perplexity is not None and perplexity > 100:
            return 3
        if quality.get("single_letter_rate", 0.0) > 0.25:
            return 3
        if quality.get("single_letter_rate", 0.0) > 0.12:
            return 1
        if quality.get("long_token_rate", 0.0) > 0.08:
            return 2
        if quality.get("space_ratio", 0.0) < 0.12:
            return 2
        return 0

    def _level0(self, text: str) -> str:
        text = _basic_normalize_text(text)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        return ' '.join(text.split())

    def _level1(self, text: str) -> str:
        text = _basic_normalize_text(text)
        text = _merge_single_letter_fragments(text)
        text = _repair_fragmented_words(text)
        text = _repair_fragmented_words_with_symspell(
            text,
            per_sentence=True,
            aggressive=True,
        )
        return ' '.join(text.split())

    def _level2(self, text: str) -> str:
        text = _basic_normalize_text(text)
        allowed_connectors = self.safe_connectors | self.risky_connectors

        def apply_split(match: re.Match) -> str:
            token = match.group(0)
            if token.lower() in self.words_dict:
                return token
            split = _split_concatenated_with_wordninja(
                token,
                self.words_dict,
                allowed_connectors,
                aggressive=True,
            )
            return split or token

        updated = re.sub(r"\b[a-zA-Z]{6,}\b", apply_split, text)
        if updated != text:
            updated = _repair_fragmented_words(updated)
        return ' '.join(updated.split())

    def _level3(self, text: str) -> str:
        candidate = self.llm.reconstruct(text)
        if candidate != text:
            return candidate
        candidate = self._level1(text)
        candidate = self._level2(candidate)
        return candidate

    def _verify(self, original: str, candidate: str) -> str:
        if candidate == original:
            return original
        if self._numbers_changed(original, candidate):
            return original
        if self._entities_lost(original, candidate):
            return original
        if self.scorer.model is not None:
            orig_ppl = self.scorer.perplexity(original)
            cand_ppl = self.scorer.perplexity(candidate)
            if orig_ppl is not None and cand_ppl is not None and cand_ppl > orig_ppl:
                return original
        return candidate

    def _numbers_changed(self, original: str, candidate: str) -> bool:
        def extract_numbers(text: str) -> List[str]:
            numbers = re.findall(r"\d+(?:[.,]\d+)*", text)
            return [n.replace(",", "") for n in numbers]

        return extract_numbers(original) != extract_numbers(candidate)

    def _entities_lost(self, original: str, candidate: str) -> bool:
        original_entities = self._capitalized_tokens(original)
        if not original_entities:
            return False
        candidate_entities = self._capitalized_tokens(candidate)
        if not candidate_entities:
            return True
        kept = len(original_entities & candidate_entities)
        return kept / max(1, len(original_entities)) < 0.6

    def _capitalized_tokens(self, text: str) -> set[str]:
        tokens = set(re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", text))
        return tokens
