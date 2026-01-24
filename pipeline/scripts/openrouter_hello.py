#!/usr/bin/env python3
"""Minimal OpenRouter sanity check: send 'hello' and expect 'world'."""

import json
import os
import sys
from typing import Any, Dict

import requests


def main() -> int:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY env var", file=sys.stderr)
        return 1

    model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return ONLY the single word: world",
            },
            {
                "role": "user",
                "content": "hello",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 5,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        print(f"OpenRouter error {response.status_code}: {response.text}", file=sys.stderr)
        return 2

    data = response.json()
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
