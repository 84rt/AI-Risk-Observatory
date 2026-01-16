from pathlib import Path

import pytest

from src.ixbrl_extractor import extract_text_from_ixbrl


def test_write_ixbrl_preview_artifact(sample_ixbrl_file):
    if sample_ixbrl_file is None:
        pytest.skip("No raw iXBRL files found.")

    report = extract_text_from_ixbrl(sample_ixbrl_file)

    output_dir = Path("data/tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ixbrl_rebuilt_preview.md"

    header = (
        "---\n"
        f"source: {sample_ixbrl_file.as_posix()}\n"
        f"spans: {len(report.spans)}\n"
        "---\n\n"
    )
    output_path.write_text(header + report.full_text, encoding="utf-8")
