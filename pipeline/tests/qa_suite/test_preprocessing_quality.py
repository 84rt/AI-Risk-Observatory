import pytest

from src.ixbrl_extractor import iXBRLExtractor
from src.preprocessor import Preprocessor, PreprocessingStrategy


def test_preprocessing_quality(sample_ixbrl_file):
    if sample_ixbrl_file is None:
        pytest.skip("No raw iXBRL files found for QA checks.")

    extractor = iXBRLExtractor()
    extracted = extractor.extract_report(sample_ixbrl_file)

    risk_preprocessor = Preprocessor(strategy=PreprocessingStrategy.RISK_ONLY)
    risk_result = risk_preprocessor.process(extracted, "QA Sample")

    keyword_preprocessor = Preprocessor(
        strategy=PreprocessingStrategy.KEYWORD,
        include_context=True
    )
    keyword_result = keyword_preprocessor.process(extracted, "QA Sample")

    assert len(risk_result.markdown_content) > 1000, "Risk-only output too small."
    assert len(keyword_result.markdown_content) > 1000, "Keyword output too small."

    risk_retention = risk_result.stats.get("retention_pct", 0)
    keyword_retention = keyword_result.stats.get("retention_pct", 0)

    assert 0 <= risk_retention <= 100, "Risk-only retention out of range."
    assert 0 <= keyword_retention <= 100, "Keyword retention out of range."
