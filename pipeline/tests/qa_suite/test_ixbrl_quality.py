import pytest

from src.ixbrl_extractor import extract_text_from_ixbrl


def _single_letter_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    single_letters = [w for w in words if len(w) == 1]
    return len(single_letters) / len(words)


def test_ixbrl_extraction_quality(sample_ixbrl_file):
    if sample_ixbrl_file is None:
        pytest.skip("No raw iXBRL files found for QA checks.")

    report = extract_text_from_ixbrl(sample_ixbrl_file)
    text = report.full_text

    assert len(text) > 20000, "Extracted text is unexpectedly small."

    words = [w for w in text.split() if w]
    assert len(words) > 3000, "Extracted text has too few tokens."

    alpha_chars = sum(c.isalpha() for c in text)
    alpha_ratio = alpha_chars / max(1, len(text))
    assert alpha_ratio > 0.4, "Extracted text has too little alphabetic content."

    single_ratio = _single_letter_ratio(words)
    assert single_ratio < 0.12, "Single-letter token ratio suggests spacing regression."

    paragraph_lengths = [
        len(span.text)
        for span in report.spans
        if not span.is_heading and span.text.strip()
    ]
    assert paragraph_lengths, "No paragraph spans detected in extracted output."
    average_length = sum(paragraph_lengths) / len(paragraph_lengths)
    assert average_length > 40, "Paragraph spans look too short on average."
