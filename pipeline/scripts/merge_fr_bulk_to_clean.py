#!/usr/bin/env python3
"""Merge multiple FR bulk datasets into one deduplicated FR_clean dataset.

Dedup policy per company-year:
1) Prefer ESEF with markdown available.
2) Else prefer non-ESEF with markdown available.
3) Else prefer ESEF without markdown.
4) Else fallback to non-ESEF without markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

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
        action="append",
        default=[],
        help="Source bulk dir with metadata.csv + markdown/. Repeat for multiple dirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "FR_clean",
        help="Output directory for merged dataset (default: data/FR_clean)",
    )
    parser.add_argument(
        "--processed-queue",
        type=Path,
        default=REPO_ROOT / "data" / "processing_queue_150.json",
        help="Queue JSON used to flag reports already processed by LLM pipeline.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize markdown files in output markdown/ (default: hardlink)",
    )
    return parser.parse_args()


def default_source_dirs() -> list[Path]:
    candidates = [
        REPO_ROOT / "data" / "FR-2021-to-2023",
        REPO_ROOT / "data" / "FR_2026-02-05 ",
        REPO_ROOT / "data" / "FR_2026-02-05",
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        if path.exists():
            out.append(path)
            seen.add(key)
    return out


def metadata_year(release_datetime: str) -> int | None:
    """Use metadata release year as the dedup year key."""
    if len(release_datetime) >= 4 and release_datetime[:4].isdigit():
        return int(release_datetime[:4])
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
    if low in {"ars", "20-f", "20-f/a"}:
        score -= 20
    return score


def processing_status_bonus(status: str) -> int:
    s = (status or "").strip().upper()
    if s in {"COMPLETED", "DONE", "SUCCEEDED", "SUCCESS"}:
        return 10
    if s in {"PENDING"}:
        return 1
    if s in {"PROCESSING"}:
        return 0
    if s in {"SKIPPED"}:
        return -4
    if s in {"FAILED"}:
        return -6
    return -1


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
    try:
        pk_int = int(str(row.get("pk") or "0"))
    except ValueError:
        pk_int = 0
    return (tier, title_score, status_score, release, pk_int)


def md_status(md_exists: bool, processing_status: str) -> str:
    if md_exists:
        return "AVAILABLE"
    status = (processing_status or "").strip()
    if status:
        return status.upper()
    return "UNKNOWN"


def load_processed_refs(queue_path: Path) -> tuple[set[str], set[tuple[str, int]]]:
    if not queue_path.exists():
        return set(), set()
    raw = json.loads(queue_path.read_text(encoding="utf-8"))
    out_pks: set[str] = set()
    out_company_year: set[tuple[str, int]] = set()
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            pk = str(row.get("pk", "")).strip()
            if pk:
                out_pks.add(pk)
            year_raw = row.get("year")
            lei = str(row.get("lei", "")).strip()
            company_name = str(row.get("company_name", "")).strip()
            if year_raw is None:
                continue
            try:
                year = int(year_raw)
            except (TypeError, ValueError):
                continue
            company_key = lei or company_name
            if company_key:
                out_company_year.add((company_key, year))
    return out_pks, out_company_year


def ensure_clean_markdown_dir(markdown_dir: Path) -> None:
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


def merge_row_values(base: dict, incoming: dict) -> dict:
    merged = dict(base)
    for key, value in incoming.items():
        if key.startswith("_"):
            continue
        if merged.get(key):
            continue
        if value:
            merged[key] = value
    # Prefer newer non-empty processing status.
    b_status = str(base.get("processing_status") or "").strip()
    i_status = str(incoming.get("processing_status") or "").strip()
    if i_status and not b_status:
        merged["processing_status"] = i_status
    return merged


def main() -> int:
    args = parse_args()
    source_dirs = [Path(p) for p in args.source_dir] if args.source_dir else default_source_dirs()
    if not source_dirs:
        raise SystemExit("No source dirs found. Pass --source-dir explicitly.")

    for d in source_dirs:
        if not d.exists():
            raise SystemExit(f"Source dir does not exist: {d}")
        if not (d / "metadata.csv").exists():
            raise SystemExit(f"Missing metadata.csv in {d}")
        if not (d / "markdown").exists():
            raise SystemExit(f"Missing markdown/ in {d}")

    processed_pks, processed_company_year = load_processed_refs(args.processed_queue)

    by_pk: dict[str, dict] = {}
    for source_dir in source_dirs:
        meta_path = source_dir / "metadata.csv"
        md_dir = source_dir / "markdown"
        with meta_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pk = str(row.get("pk") or "").strip()
                if not pk:
                    continue

                row = dict(row)
                row["_source_dirs"] = [str(source_dir)]
                row["_md_paths"] = []
                md_path = md_dir / f"{pk}.md"
                if md_path.exists():
                    row["_md_paths"].append(str(md_path))

                if pk not in by_pk:
                    by_pk[pk] = row
                else:
                    existing = by_pk[pk]
                    merged = merge_row_values(existing, row)
                    sources = sorted(set(existing.get("_source_dirs", [])) | set(row.get("_source_dirs", [])))
                    md_paths = sorted(set(existing.get("_md_paths", [])) | set(row.get("_md_paths", [])))
                    merged["_source_dirs"] = sources
                    merged["_md_paths"] = md_paths
                    by_pk[pk] = merged

    rows: list[dict] = []
    for pk, row in by_pk.items():
        release = str(row.get("release_datetime") or "")
        dedup_year = metadata_year(release)
        if dedup_year is None:
            continue

        company_name = str(row.get("company__name") or "").strip()
        lei = str(row.get("company__lei") or "").strip()
        company_key = lei or company_name
        filing_type = str(row.get("filing_type__name") or "").lower()
        is_esef = "esef" in filing_type
        md_paths = row.get("_md_paths", [])
        chosen_md_path = md_paths[0] if md_paths else ""
        has_md = bool(chosen_md_path)

        row["_report_year"] = dedup_year
        row["_company_key"] = company_key
        row["_is_esef"] = is_esef
        row["_has_md"] = has_md
        row["_chosen_md_path"] = chosen_md_path
        row["_score"] = selection_score(row, md_exists=has_md, is_esef=is_esef)
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
                    "markdown_exists": "yes" if row.get("_has_md") else "no",
                    "source_datasets": ";".join(row.get("_source_dirs", [])),
                }
            )

    selected.sort(
        key=lambda r: (
            str(r.get("company__name", "")).lower(),
            int(r["_report_year"]),
            str(r.get("release_datetime", "")),
        )
    )

    output_dir = args.output_dir
    output_markdown_dir = output_dir / "markdown"
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_markdown_dir(output_markdown_dir)

    output_rows: list[dict] = []
    missing_rows: list[dict] = []
    for row in selected:
        pk = str(row.get("pk", "")).strip()
        chosen_md_path = str(row.get("_chosen_md_path") or "")
        has_md = bool(chosen_md_path)

        if has_md:
            src = Path(chosen_md_path)
            dst = output_markdown_dir / f"{pk}.md"
            materialize_markdown(src, dst, args.link_mode)
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
                    "md_status": md_status(False, str(row.get("processing_status", ""))),
                    "dedup_report_year": row.get("_report_year"),
                    "source_datasets": ";".join(row.get("_source_dirs", [])),
                }
            )

        out_row = {
            **{k: v for k, v in row.items() if not str(k).startswith("_")},
            "dedup_report_year": row.get("_report_year"),
            "markdown_exists": "yes" if has_md else "no",
            "md_status": md_status(has_md, str(row.get("processing_status", ""))),
            "source_datasets": ";".join(row.get("_source_dirs", [])),
            "llm_processed_150": "yes" if pk in processed_pks else "no",
            "llm_processed_150_company_year": (
                "yes"
                if (str(row.get("company__lei") or row.get("company__name") or "").strip(), int(row.get("_report_year")))
                in processed_company_year
                else "no"
            ),
            "selection_policy": "ESEF-first-markdown-first",
            "dedup_year_basis": "release_datetime",
        }
        output_rows.append(out_row)

    metadata_fields = [
        "pk",
        "company__name",
        "company__ticker",
        "company__lei",
        "company__country_iso__name",
        "title",
        "filing_type__name",
        "release_datetime",
        "language__name",
        "source__name",
        "source_filing_id",
        "processing_status",
        "added_to_platform",
        "dedup_report_year",
        "markdown_exists",
        "md_status",
        "source_datasets",
        "llm_processed_150",
        "llm_processed_150_company_year",
        "selection_policy",
        "dedup_year_basis",
    ]
    write_csv(output_dir / "metadata.csv", metadata_fields, output_rows)

    dropped_fields = [
        "pk",
        "company__name",
        "company__lei",
        "title",
        "filing_type__name",
        "release_datetime",
        "processing_status",
        "dedup_report_year",
        "markdown_exists",
        "source_datasets",
    ]
    write_csv(output_dir / "dropped_candidates.csv", dropped_fields, dropped)

    missing_fields = [
        "pk",
        "company__name",
        "company__lei",
        "title",
        "filing_type__name",
        "release_datetime",
        "processing_status",
        "md_status",
        "dedup_report_year",
        "source_datasets",
    ]
    write_csv(output_dir / "missing_markdown_after_dedup.csv", missing_fields, missing_rows)

    md_status_counts = Counter(row["md_status"] for row in output_rows)
    summary = {
        "source_dirs": [str(p) for p in source_dirs],
        "output_dir": str(output_dir),
        "distinct_filing_ids_after_union": len(by_pk),
        "deduplicated_rows": len(output_rows),
        "dropped_rows": len(dropped),
        "markdown_files_materialized": len(output_rows) - len(missing_rows),
        "missing_markdown_after_dedup": len(missing_rows),
        "md_status_counts": dict(sorted(md_status_counts.items())),
        "llm_processed_150_yes": sum(1 for r in output_rows if r["llm_processed_150"] == "yes"),
        "llm_processed_150_no": sum(1 for r in output_rows if r["llm_processed_150"] == "no"),
        "llm_processed_150_company_year_yes": sum(
            1 for r in output_rows if r["llm_processed_150_company_year"] == "yes"
        ),
        "llm_processed_150_company_year_no": sum(
            1 for r in output_rows if r["llm_processed_150_company_year"] == "no"
        ),
        "link_mode": args.link_mode,
        "dedup_year_basis": "release_datetime",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Merged FR_clean build complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
