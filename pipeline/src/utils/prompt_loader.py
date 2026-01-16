"""Prompt loader utilities for classifier prompts stored in YAML."""

from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml

from ..config import get_settings


@lru_cache(maxsize=1)
def _load_prompt_yaml() -> Dict[str, str]:
    """Load classifier prompts from YAML."""
    settings = get_settings()
    prompt_path = settings.pipeline_root / "prompts" / "classifiers.yaml"
    data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Prompt YAML must contain a top-level mapping")
    return data


def get_prompt_template(key: str) -> str:
    """Return the prompt template string for the given key."""
    prompts = _load_prompt_yaml()
    if key not in prompts:
        raise KeyError(f"Prompt key not found: {key}")
    template = prompts[key]
    if not isinstance(template, str):
        raise ValueError(f"Prompt template for {key} must be a string")
    return template
