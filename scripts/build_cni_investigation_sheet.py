#!/usr/bin/env python3
"""Build a human review sheet for inconsistent CNI mappings within an ISIC code."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPANY_CNI_CSV = REPO_ROOT / "data" / "reference" / "company_cni_sectors.csv"
FR_COMPANIES_JSON = REPO_ROOT / "data" / "FR_dataset" / "companies.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "reference" / "cni_investigation_sheet.csv"


OUT_FIELDS = [
    "review_group_rank",
    "review_company_rank",
    "review_priority",
    "isic_code",
    "isic_name",
    "isic_company_count",
    "isic_sector_breakdown",
    "majority_primary_sector",
    "majority_primary_pct",
    "could_promote_static",
    "matches_majority",
    "company_name",
    "lei",
    "assigned_primary_sector",
    "assigned_sectors",
    "source",
    "fr_tagline",
    "fr_description",
    "reviewer_status",
    "reviewer_decision",
    "reviewer_corrected_primary",
    "reviewer_corrected_sectors",
    "reviewer_notes",
]


def load_fr_metadata() -> dict[str, dict[str, str]]:
    with FR_COMPANIES_JSON.open() as f:
        raw = json.load(f)

    result: dict[str, dict[str, str]] = {}
    for company in raw:
        lei = company.get("lei")
        if not lei:
            continue
        result[lei] = {
            "tagline": company.get("tagline") or "",
            "description": company.get("description") or "",
        }
    return result


def load_company_rows(source: str) -> list[dict[str, str]]:
    with COMPANY_CNI_CSV.open() as f:
        rows = [row for row in csv.DictReader(f) if row.get("source") == source]
    return rows


def collect_inconsistent_groups(
    rows: list[dict[str, str]],
    selected_isic: set[str] | None,
    min_group_size: int,
) -> list[dict]:
    by_isic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        code = (row.get("isic_code") or "").strip()
        if not code:
            continue
        if selected_isic and code not in selected_isic:
            continue
        by_isic[code].append(row)

    groups: list[dict] = []
    for code, recs in by_isic.items():
        if len(recs) < min_group_size:
            continue

        counts = Counter(
            row["cni_sector_primary"] for row in recs if row.get("cni_sector_primary")
        )
        if len(counts) <= 1:
            continue

        majority_sector, majority_count = counts.most_common(1)[0]
        groups.append(
            {
                "isic_code": code,
                "isic_name": recs[0].get("isic_name", ""),
                "records": recs,
                "counts": counts,
                "majority_sector": majority_sector,
                "majority_pct": majority_count / len(recs) * 100,
            }
        )

    groups.sort(key=lambda g: (-len(g["records"]), g["isic_code"]))
    return groups


def build_review_rows(
    groups: list[dict],
    fr_metadata: dict[str, dict[str, str]],
    static_threshold: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for group_rank, group in enumerate(groups, start=1):
        counts: Counter = group["counts"]
        majority_sector = group["majority_sector"]
        majority_pct = group["majority_pct"]
        breakdown = ", ".join(f"{sector}:{count}" for sector, count in counts.most_common())
        could_promote_static = "yes" if majority_pct >= static_threshold else "no"

        sorted_records = sorted(
            group["records"],
            key=lambda row: (
                row.get("cni_sector_primary") == majority_sector,
                row.get("company_name", ""),
            ),
        )

        for company_rank, row in enumerate(sorted_records, start=1):
            lei = row.get("lei", "")
            fr = fr_metadata.get(lei, {})
            matches_majority = row.get("cni_sector_primary") == majority_sector

            rows.append(
                {
                    "review_group_rank": str(group_rank),
                    "review_company_rank": str(company_rank),
                    "review_priority": "1" if not matches_majority else "2",
                    "isic_code": group["isic_code"],
                    "isic_name": group["isic_name"],
                    "isic_company_count": str(len(group["records"])),
                    "isic_sector_breakdown": breakdown,
                    "majority_primary_sector": majority_sector,
                    "majority_primary_pct": f"{majority_pct:.1f}",
                    "could_promote_static": could_promote_static,
                    "matches_majority": "yes" if matches_majority else "no",
                    "company_name": row.get("company_name", ""),
                    "lei": lei,
                    "assigned_primary_sector": row.get("cni_sector_primary", ""),
                    "assigned_sectors": row.get("cni_sectors", ""),
                    "source": row.get("source", ""),
                    "fr_tagline": fr.get("tagline", ""),
                    "fr_description": fr.get("description", ""),
                    "reviewer_status": "",
                    "reviewer_decision": "",
                    "reviewer_corrected_primary": "",
                    "reviewer_corrected_sectors": "",
                    "reviewer_notes": "",
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a human review sheet for inconsistent LLM-assigned CNI sectors."
    )
    ap.add_argument(
        "--source",
        default="llm_gemini",
        help="Row source to review (default: llm_gemini).",
    )
    ap.add_argument(
        "--isic",
        action="append",
        default=[],
        help="Restrict to a specific ISIC code. Repeatable.",
    )
    ap.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum companies required in an ISIC group (default: 2).",
    )
    ap.add_argument(
        "--limit-groups",
        type=int,
        default=0,
        help="Optional limit on number of inconsistent ISIC groups to export.",
    )
    ap.add_argument(
        "--static-threshold",
        type=float,
        default=80.0,
        help="Majority percentage at which a group is marked as a static-rule candidate.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    selected_isic = {code.strip() for code in args.isic if code.strip()} or None

    company_rows = load_company_rows(args.source)
    fr_metadata = load_fr_metadata()
    groups = collect_inconsistent_groups(
        company_rows,
        selected_isic=selected_isic,
        min_group_size=args.min_group_size,
    )

    if args.limit_groups > 0:
        groups = groups[: args.limit_groups]

    review_rows = build_review_rows(
        groups=groups,
        fr_metadata=fr_metadata,
        static_threshold=args.static_threshold,
    )
    write_csv(args.output, review_rows)

    print(
        f"Wrote {len(review_rows)} review rows across {len(groups)} inconsistent ISIC groups "
        f"to {args.output.relative_to(REPO_ROOT)}"
    )
    for group in groups[:20]:
        counts = ", ".join(
            f"{sector}:{count}" for sector, count in group["counts"].most_common()
        )
        print(
            f"{group['isic_code']:<6} {len(group['records']):>5}  "
            f"{group['isic_name'][:40]:<40}  {counts}"
        )


if __name__ == "__main__":
    main()
