#!/usr/bin/env python3
"""Minimal Gemini API probe for a small set of model names."""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from google import genai
from google.genai import types


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "gemini-3.1-preview",
    "gemini-3",
    "gemini-3-flash",
]


def load_repo_env() -> None:
    """Load repo-local env files without depending on pipeline settings."""
    load_dotenv(REPO_ROOT / ".env.local", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in env/.env.local")
    return api_key


def short_name(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name.split("/", 1)[1]
    return model_name


def list_model_names(client: genai.Client) -> list[str]:
    models: list[str] = []
    for model in client.models.list():
        name = getattr(model, "name", None)
        if name:
            models.append(short_name(name))
    return sorted(set(models))


def related_models(query: str, available: Iterable[str], limit: int = 8) -> list[str]:
    names = list(available)
    lowered = query.lower()

    contains = [name for name in names if lowered in name.lower()]
    if contains:
        return contains[:limit]

    token = lowered.replace(".", "")
    loose = [name for name in names if token in name.lower().replace(".", "")]
    if loose:
        return loose[:limit]

    return difflib.get_close_matches(query, names, n=limit, cutoff=0.35)


def probe_model(client: genai.Client, model_name: str) -> tuple[bool, str]:
    response = client.models.generate_content(
        model=model_name,
        contents="Reply with exactly: OK",
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=32,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        parts: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text.strip())
        text = "\n".join(part for part in parts if part).strip()

    finish_reasons: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            finish_reasons.append(str(reason))

    suffix = f" | finish_reason={', '.join(finish_reasons)}" if finish_reasons else ""
    return True, f"{text}{suffix}" if text else suffix.lstrip(" |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model name to probe. Repeat for multiple models.",
    )
    parser.add_argument(
        "--show-matches",
        type=int,
        default=8,
        help="How many related model names to print per probe (default: 8).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print matching Gemini model names from the API.",
    )
    args = parser.parse_args()

    load_repo_env()

    try:
        client = genai.Client(api_key=get_api_key())
    except Exception as exc:
        print(f"ENV ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        available = list_model_names(client)
    except Exception as exc:
        print(f"LIST ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    targets = args.models or DEFAULT_MODELS
    gemini_models = [name for name in available if name.startswith("gemini")]

    print(f"Loaded {len(available)} models from Gemini API.")
    print()

    if args.list_only:
        for target in targets:
            print(f"[matches] {target}")
            matches = related_models(target, gemini_models, limit=args.show_matches)
            if matches:
                for name in matches:
                    print(f"  - {name}")
            else:
                print("  - none")
            print()
        return 0

    any_failures = False
    for target in targets:
        exact = target in available
        print(f"[probe] {target}")
        print(f"  listed_by_api: {'yes' if exact else 'no'}")

        matches = related_models(target, gemini_models, limit=args.show_matches)
        if matches:
            print(f"  related_models: {', '.join(matches)}")
        else:
            print("  related_models: none")

        try:
            _, text = probe_model(client, target)
            print("  generate_content: ok")
            print(f"  response: {text!r}")
        except Exception as exc:
            any_failures = True
            print(f"  generate_content: fail ({type(exc).__name__})")
            print(f"  error: {exc}")

        print()

    return 1 if any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
