"""Helpers for loading company reference data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

from .identifiers import make_company_id


def _normalize_company_number(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if cleaned.isdigit():
        return cleaned.zfill(8)
    return cleaned


def load_companies_csv(path: Path) -> List[Dict[str, Optional[str]]]:
    """Load companies from a CSV into canonical keys."""
    companies: List[Dict[str, Optional[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company_name = (row.get("company_name") or row.get("name") or row.get("company") or "").strip()
            sector = (row.get("sector") or row.get("cni_sector") or "Unknown").strip()
            index_name = (row.get("index") or "").strip()
            company_type = (row.get("type") or "").strip()

            company_number = _normalize_company_number(row.get("company_number"))
            lei = (row.get("lei") or "").strip() or None
            ticker = (row.get("ticker") or "").strip() or None

            company_id = make_company_id(company_number, lei, company_name)

            companies.append(
                {
                    "company_id": company_id,
                    "company_name": company_name,
                    "company_number": company_number,
                    "ticker": ticker,
                    "lei": lei,
                    "sector": sector,
                    "index": index_name,
                    "type": company_type,
                }
            )
    return companies
