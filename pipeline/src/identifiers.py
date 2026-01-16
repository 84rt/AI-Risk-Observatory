"""Identifier helpers for companies and documents."""

from __future__ import annotations

from typing import Optional


def slugify(value: Optional[str]) -> str:
    """Minimal slugifier for IDs."""
    if not value:
        return ""
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def make_company_id(
    company_number: Optional[str],
    lei: Optional[str],
    company_name: Optional[str],
) -> str:
    """Create a stable company identifier for storage."""
    if company_number:
        return company_number
    if lei:
        return lei
    slug = slugify(company_name)
    return slug or "unknown-company"


def make_document_id(
    company_number: Optional[str],
    lei: Optional[str],
    company_name: Optional[str],
    year: int,
    sector: Optional[str],
    fmt: str,
) -> str:
    """Stable document_id based on company + year + sector + format."""
    identifier = make_company_id(company_number, lei, company_name)
    return f"{identifier}-{year}-{slugify(sector or 'unknown')}-{fmt}"
