from src.ixbrl_extractor import iXBRLParser


def test_ixbrl_clean_text_keeps_articles():
    parser = iXBRLParser()
    text = "management uses a number of alternative measures"
    assert parser._clean_text(text) == text


def test_ixbrl_clean_text_merges_split_suffix():
    parser = iXBRLParser()
    assert parser._clean_text("principal risk s and uncertainties") == "principal risks and uncertainties"


def test_ixbrl_clean_text_merges_split_prefix():
    parser = iXBRLParser()
    assert parser._clean_text("the c ustomer base") == "the customer base"


def test_ixbrl_clean_text_keeps_article_prefix():
    parser = iXBRLParser()
    text = "a company overview"
    assert parser._clean_text(text) == text
