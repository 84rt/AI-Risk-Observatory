"""Prompt loader utilities for classifier prompts stored in YAML."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

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
    """Return the system prompt template for the given key."""
    prompts = _load_prompt_yaml()
    if key not in prompts:
        raise KeyError(f"Prompt key not found: {key}")
    template = prompts[key]
    if not isinstance(template, str):
        raise ValueError(f"Prompt template for {key} must be a string")
    return template


_DEFAULT_USER_TEMPLATE = """## EXCERPT
\"\"\"
{text}
\"\"\"
"""

_REASONING_POLICIES = {
    "none": {
        "reasoning_instruction": "Do NOT include a reasoning field in the JSON.",
        "reasoning_field": "",
    },
    "short": {
        "reasoning_instruction": "Include a \"reasoning\" field with a brief explanation.",
        "reasoning_field": ", \"reasoning\": \"Brief explanation\"",
    },
    "limited": {
        "reasoning_instruction": "Include a \"reasoning\" field (<= 200 words).",
        "reasoning_field": ", \"reasoning\": \"Up to a few sentences (<= 200 words).\"",
    },
}


def get_prompt_messages(
    key: str,
    *,
    reasoning_policy: str = "short",
    user_template: str = _DEFAULT_USER_TEMPLATE,
    **kwargs: str,
) -> Tuple[str, str]:
    """Render prompt templates and return (system, user) strings."""
    template = get_prompt_template(key)
    policy = _REASONING_POLICIES.get(reasoning_policy)
    if policy is None:
        raise ValueError(
            f"Unknown reasoning_policy: {reasoning_policy}. "
            f"Valid: {sorted(_REASONING_POLICIES.keys())}"
        )
    system = template.format(
        reasoning_instruction=policy["reasoning_instruction"],
        reasoning_field=policy["reasoning_field"],
        **kwargs,
    )
    user = user_template.format(**kwargs)
    return system, user
