"""Mistral OCR: convert a PDF to markdown via the Mistral OCR API.

Public interface:

    from src.mistral_ocr import process_pdf_to_markdown

    markdown = process_pdf_to_markdown(
        pdf_path,
        api_key=...,          # MISTRAL_API_KEY
        model="mistral-ocr-latest",
        timeout=300,
        extract_header=True,
        extract_footer=True,
    )

The function sends the entire PDF as a base64-encoded data URL in a single
request and stitches the per-page markdown blocks into one document.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

import requests

MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
DEFAULT_MODEL = "mistral-ocr-latest"
DEFAULT_TIMEOUT = 300


def process_pdf_to_markdown(
    pdf_path: Path,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    extract_header: bool = True,
    extract_footer: bool = True,
) -> str:
    """Send a PDF to Mistral OCR and return the stitched markdown.

    Args:
        pdf_path: Path to the input PDF file.
        api_key: Mistral API key (``MISTRAL_API_KEY``).
        model: Mistral OCR model name.
        timeout: HTTP timeout in seconds.
        extract_header: Ask the OCR API to extract page headers separately.
        extract_footer: Ask the OCR API to extract page footers separately.

    Returns:
        Full markdown string with all pages stitched together.

    Raises:
        requests.HTTPError: If the Mistral API returns a non-2xx status.
        requests.RequestException: On connection / timeout errors.
    """
    pdf_bytes = Path(pdf_path).read_bytes()
    data_url = _bytes_as_data_url(pdf_bytes)
    payload = _build_payload(data_url, model=model, extract_header=extract_header, extract_footer=extract_footer)
    response = _call_ocr_api(api_key, payload, timeout)
    return _stitch_markdown(response.get("pages") or [])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bytes_as_data_url(pdf_bytes: bytes) -> str:
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def _build_payload(
    document_data_url: str,
    *,
    model: str,
    extract_header: bool,
    extract_footer: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "document": {
            "type": "document_url",
            "document_url": document_data_url,
        },
        "include_image_base64": False,
    }
    if extract_header:
        payload["extract_header"] = True
    if extract_footer:
        payload["extract_footer"] = True
    return payload


def _call_ocr_api(
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    response = requests.post(
        MISTRAL_OCR_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _stitch_markdown(pages: list[dict[str, Any]]) -> str:
    ordered = sorted(pages, key=lambda page: int(page.get("index", 0)))
    chunks: list[str] = []
    for page in ordered:
        text = str(page.get("markdown") or "").strip()
        if text:
            chunks.append(text)
    return ("\n\n".join(chunks)).strip() + "\n"


def strip_images(text: str) -> str:
    """Remove markdown image syntax and <img> tags."""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<img\b[^>]*>", "", text, flags=re.IGNORECASE)
    return text
