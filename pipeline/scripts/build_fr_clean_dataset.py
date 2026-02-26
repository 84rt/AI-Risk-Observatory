#!/usr/bin/env python3
"""Build a deduplicated FR dataset with ESEF-first selection.

Policy per company-year:
1) Prefer ESEF filing with local markdown available.
2) Otherwise prefer non-ESEF filing with local markdown available.
3) Otherwise prefer ESEF filing (missing markdown).
4) Otherwise fallback to non-ESEF filing (missing markdown).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "FR-2021-to-2023"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "FR_clean"

RANGE_RE = re.compile(r"(20\d{2})\s*[/\-]\s*(\d{2}|20\d{2})")
YEAR_RE = re.compile(r"\b(20\d{2})\b")
BAD_TITLE_TOKENS = (
    "20-f",
    "20-f/a",
    "ars",
    "notice of agm",
    "replacement financial statements",
    "preliminary",
    "proxy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Input FR bulk directory containing metadata.csv and markdown/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for deduplicated dataset",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize markdown files in output markdown/ (default: hardlink)",
    )
    return parser.parse_args()


def infer_report_year(title: str, release_datetime: str) -> int | None:
    title = title or ""
    release_year = None
    if len(release_datetime) >= 4 and release_datetime[:4].isdigit():
        release_year = int(release_datetime[:4])

    range_ends: list[int] = []
    for start_raw, end_raw in RANGE_RE.findall(title):
        start = int(start_raw)
        if len(end_raw) == 2:
            end = (start // 100) * 100 + int(end_raw)
        else:
            end = int(end_raw)
        range_ends.append(end)
    if range_ends:
        return max(range_ends)

    years = [int(y) for y in YEAR_RE.findall(title)]
    if years:
        return max(years)

    # Common annual-report convention.
    if release_year is not None:
        return release_year - 1
    return None


def title_quality(title: str) -> int:
    low = (title or "").lower().strip()
    score = 0
    if "annual report and accounts" in low:
        score += 60
    elif "annual report" in low:
        score += 45
    elif "annual" in low and "report" in low:
        score += 30
    if "esef" in low:
        score += 10

    for token in BAD_TITLE_TOKENS:
        if token in low:
            score -= 25

    # Exact short labels are usually less descriptive.
    if low in {"ars", "20-f", "20-f/a"}:
        score -= 20
    return score


def processing_status_bonus(status: str) -> int:
    s = (status or "").strip().upper()
    if s in {"COMPLETED", "DONE", "SUCCEEDED", "SUCCESS"}:
        return 8
    if s in {"PENDING", "PROCESSING"}:
        return -2
    if s in {"FAILED", "SKIPPED"}:
        return -6
    return 0


def selection_score(row: dict, md_exists: bool, is_esef: bool) -> tuple[int, int, int, str, int]:
    if md_exists and is_esef:
        tier = 4
    elif md_exists and not is_esef:
        tier = 3
    elif is_esef:
        tier = 2
    else:
        tier = 1

    title_score = title_quality(row.get("title", ""))
    status_score = processing_status_bonus(row.get("processing_status", ""))
    release = str(row.get("release_datetime", ""))
    pk = int(str(row.get("pk") or "0"))
    return tier, title_score, status_score, release, pk


def ensure_empty_markdown_dir(markdown_dir: Path) -> None:
    markdown_dir.mkdir(parents=True, exist_ok=True)
    for path in markdown_dir.glob("*.md"):
        path.unlink()


def materialize_markdown(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir

    metadata_path = source_dir / "metadata.csv"
    source_markdown_dir = source_dir / "markdown"
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata.csv at {metadata_path}")
    if not source_markdown_dir.exists():
        raise SystemExit(f"Missing markdown dir at {source_markdown_dir}")

    rows: list[dict] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pk = (row.get("pk") or "").strip()
            if not pk:
                continue
            report_year = infer_report_year(row.get("title", ""), row.get("release_datetime", ""))
            if report_year is None:
                continue
            lei = (row.get("company__lei") or "").strip()
            company_name = (row.get("company__name") or "").strip()
            company_key = lei or company_name
            filing_type = (row.get("filing_type__name") or "").lower()
            is_esef = "esef" in filing_type
            md_exists = (source_markdown_dir / f"{pk}.md").exists()

            row = dict(row)
            row["_report_year"] = report_year
            row["_company_key"] = company_key
            row["_is_esef"] = is_esef
            row["_md_exists"] = md_exists
            row["_score"] = selection_score(row, md_exists=md_exists, is_esef=is_esef)
            rows.append(row)

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["_company_key"], row["_report_year"])].append(row)

    selected: list[dict] = []
    dropped: list[dict] = []
    for key, candidates in groups.items():
        best = max(candidates, key=lambda r: r["_score"])
        selected.append(best)
        for row in candidates:
            if row is best:
                continue
            dropped.append(
                {
                    "pk": row.get("pk", ""),
                    "company__name": row.get("company__name", ""),
                    "company__lei": row.get("company__lei", ""),
                    "title": row.get("title", ""),
                    "filing_type__name": row.get("filing_type__name", ""),
                    "release_datetime": row.get("release_datetime", ""),
                    "processing_status": row.get("processing_status", ""),
                    "dedup_report_year": row.get("_report_year"),
                }
            )

    selected.sort(
        key=lambda r: (
            str(r.get("company__name", "")).lower(),
            int(r["_report_year"]),
            str(r.get("release_datetime", "")),
        )
    )

    output_markdown_dir = output_dir / "markdown"
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_empty_markdown_dir(output_markdown_dir)

    selected_rows: list[dict] = []
    missing_rows: list[dict] = []
    for row in selected:
        pk = str(row["pk"]).strip()
        src_md = source_markdown_dir / f"{pk}.md"
        dst_md = output_markdown_dir / f"{pk}.md"
        md_exists = src_md.exists()

        if md_exists:
            materialize_markdown(src_md, dst_md, mode=args.link_mode)
        else:
            missing_rows.append(
                {
                    "pk": pk,
                    "company__name": row.get("company__name", ""),
                    "company__lei": row.get("company__lei", ""),
                    "title": row.get("title", ""),
                    "filing_type__name": row.get("filing_type__name", ""),
                    "release_datetime": row.get("release_datetime", ""),
                    "processing_status": row.get("processing_status", ""),
                    "dedup_report_year": row.get("_report_year"),
                }
            )

        selected_rows.append(
            {
                **{k: v for k, v in row.items() if not str(k).startswith("_")},
                "dedup_report_year": row.get("_report_year"),
                "markdown_exists": "yes" if md_exists else "no",
                "selection_policy": "ESEF-first-markdown-first",
            }
        )

    # Persist outputs.
    selected_fieldnames = list(selected_rows[0].keys()) if selected_rows else []
    write_csv(output_dir / "metadata.csv", selected_fieldnames, selected_rows)

    if dropped:
        write_csv(output_dir / "dropped_candidates.csv", dropped[0].keys(), dropped)
    else:
        write_csv(
            output_dir / "dropped_candidates.csv",
            [
                "pk",
                "company__name",
                "company__lei",
                "title",
                "filing_type__name",
                "release_datetime",
                "processing_status",
                "dedup_report_year",
            ],
            [],
        )

    if missing_rows:
        write_csv(output_dir / "missing_markdown_after_dedup.csv", missing_rows[0].keys(), missing_rows)
    else:
        write_csv(
            output_dir / "missing_markdown_after_dedup.csv",
            [
                "pk",
                "company__name",
                "company__lei",
                "title",
                "filing_type__name",
                "release_datetime",
                "processing_status",
                "dedup_report_year",
            ],
            [],
        )

    source_manifest_pdf = source_dir / "FinancialReports_Manifest.pdf"
    if source_manifest_pdf.exists():
        shutil.copy2(source_manifest_pdf, output_dir / "FinancialReports_Manifest.pdf")

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "rows_in_source_metadata": len(rows),
        "deduplicated_rows": len(selected_rows),
        "dropped_rows": len(dropped),
        "markdown_files_copied_or_linked": len(selected_rows) - len(missing_rows),
        "missing_markdown_after_dedup": len(missing_rows),
        "link_mode": args.link_mode,
        "selection_policy": "ESEF-first + markdown-availability-first fallback",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("FR_clean build complete")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
