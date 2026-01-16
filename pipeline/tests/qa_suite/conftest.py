import sys
from pathlib import Path

import pytest


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT))


@pytest.fixture(scope="session")
def sample_ixbrl_file() -> Path | None:
    ixbrl_dir = PIPELINE_ROOT.parent / "data" / "raw" / "ixbrl"
    if not ixbrl_dir.exists():
        return None
    ixbrl_files = sorted(ixbrl_dir.rglob("*.xhtml"))
    if not ixbrl_files:
        return None
    return ixbrl_files[0]
