import csv
import json
import re
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Paths
manifest_path = Path("data/reference/companies_metadata_v2.csv")
fr_metadata_path = Path("data/FinancialReports_downloaded/metadata.csv")
fr_markdown_dir = Path("data/FinancialReports_downloaded/markdown")
output_path = Path("data/processing_queue.json")

DEFAULT_TARGET_YEARS = (2023, 2024)

RANGE_RE = re.compile(r"(20\d{2})\s*[/\-]\s*(\d{2}|\d{4})")
YEAR_RE = re.compile(r"\b(20\d{2})\b")
YEAR_ENDED_RE = re.compile(r"(?:year\s+ended|for\s+the\s+year\s+ended)\D{0,40}(20\d{2})", re.IGNORECASE)
NOISE_TITLE_RE = re.compile(
    r"\b(preliminary|notice|proxy|transaction|holding\(s\)|director/pdmr|results of annual general meeting|agm)\b",
    re.IGNORECASE,
)

# Signal priorities for mapping filings to dataset years.
SIGNAL_WEIGHTS = {
    "title_range_end": 500,
    "title_year_ended": 450,
    "title_year_only": 400,
    "release_minus_one": 320,
    "release_year": 220,
}


@dataclass
class Candidate:
    row: dict
    pk: str
    release_date: str
    title: str
    filing_type: str
    md_exists: bool
    is_esef: bool
    is_annual_type: bool
    is_annual_title: bool
    is_noise_title: bool
    title_years: set[int]
    signals: dict[int, int]  # target_year -> signal_weight


def _parse_release_year(row: dict) -> int | None:
    release = str(row.get("release_datetime") or "").strip()
    if len(release) >= 4 and release[:4].isdigit():
        return int(release[:4])
    return None


def _parse_title_year_signals(title: str, target_years: set[int]) -> dict[int, int]:
    signals: dict[int, int] = {}

    # Prefer explicit fiscal ranges like 2023/24 -> 2024.
    for y1, y2 in RANGE_RE.findall(title):
        start = int(y1)
        end = int(y2) if len(y2) == 4 else (start // 100) * 100 + int(y2)
        if end in target_years:
            signals[end] = max(signals.get(end, 0), SIGNAL_WEIGHTS["title_range_end"])
        if start in target_years:
            # Weaker than range-end, but still useful.
            signals[start] = max(signals.get(start, 0), SIGNAL_WEIGHTS["title_year_only"] - 50)

    # Phrases like "for the year ended ... 2023"
    for y in YEAR_ENDED_RE.findall(title):
        year = int(y)
        if year in target_years:
            signals[year] = max(signals.get(year, 0), SIGNAL_WEIGHTS["title_year_ended"])

    # If title contains a single target year, it's often reliable.
    title_years = sorted({int(y) for y in YEAR_RE.findall(title) if int(y) in target_years})
    if len(title_years) == 1:
        year = title_years[0]
        signals[year] = max(signals.get(year, 0), SIGNAL_WEIGHTS["title_year_only"])

    return signals


def _build_candidate(row: dict, target_years: set[int]) -> Candidate | None:
    pk = str(row.get("pk") or "").strip()
    if not pk:
        return None

    title = str(row.get("title") or "")
    filing_type = str(row.get("filing_type__name") or "")
    release_year = _parse_release_year(row)

    signals = _parse_title_year_signals(title, target_years)
    if release_year is not None:
        prev = release_year - 1
        if prev in target_years:
            signals[prev] = max(signals.get(prev, 0), SIGNAL_WEIGHTS["release_minus_one"])
        if release_year in target_years:
            signals[release_year] = max(signals.get(release_year, 0), SIGNAL_WEIGHTS["release_year"])

    if not signals:
        return None

    md_path = fr_markdown_dir / f"{pk}.md"
    filing_type_lower = filing_type.lower()
    title_lower = title.lower()

    title_years = {int(y) for y in YEAR_RE.findall(title)}

    return Candidate(
        row=row,
        pk=pk,
        release_date=str(row.get("release_datetime") or ""),
        title=title,
        filing_type=filing_type,
        md_exists=md_path.exists(),
        is_esef="esef" in filing_type_lower,
        is_annual_type="annual report" in filing_type_lower,
        is_annual_title="annual" in title_lower and "report" in title_lower,
        is_noise_title=bool(NOISE_TITLE_RE.search(title)),
        title_years=title_years,
        signals=signals,
    )


def _candidate_score(candidate: Candidate, year: int, target_years_set: set[int]) -> tuple[int, str, str]:
    signal = candidate.signals.get(year, -1)
    score = signal
    if candidate.md_exists:
        score += 1000
    if candidate.is_esef:
        score += 120
    if candidate.is_annual_type:
        score += 80
    if candidate.is_annual_title:
        score += 40
    if candidate.is_noise_title:
        score -= 240
    # If we only reached this year via release-year fallback but title explicitly
    # points to another year, down-rank this candidate.
    if (
        signal == SIGNAL_WEIGHTS["release_minus_one"]
        and candidate.title_years
        and year not in candidate.title_years
        and any(y in target_years_set and y != year for y in candidate.title_years)
    ):
        score -= 260
    return score, candidate.release_date, candidate.pk


def _select_best_by_year(candidates: list[Candidate], target_years: tuple[int, ...]) -> dict[int, Candidate]:
    ranked: dict[int, list[Candidate]] = {}
    target_years_set = set(target_years)
    for year in target_years:
        pool = [c for c in candidates if year in c.signals]
        pool.sort(key=lambda c: _candidate_score(c, year, target_years_set), reverse=True)
        ranked[year] = pool

    selected: dict[int, Candidate] = {}
    used_pks: set[str] = set()

    # Fill the hardest year first (fewest candidates), to avoid collisions.
    order = sorted(target_years, key=lambda y: len(ranked.get(y, [])))
    for year in order:
        pool = ranked.get(year, [])
        if not pool:
            continue
        chosen = next((c for c in pool if c.pk not in used_pks), pool[0])
        selected[year] = chosen
        used_pks.add(chosen.pk)

    return selected


def _parse_years_csv(years_csv: str) -> tuple[int, ...]:
    years = []
    for token in years_csv.split(","):
        tok = token.strip()
        if not tok:
            continue
        if not tok.isdigit() or len(tok) != 4:
            raise ValueError(f"Invalid year token: {tok!r}")
        years.append(int(tok))
    if not years:
        raise ValueError("No years provided")
    return tuple(sorted(set(years)))


def generate_queue(target_years: tuple[int, ...] = DEFAULT_TARGET_YEARS, output_file: Path = output_path):
    # 1) Load company manifest.
    companies: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lei = (row.get("lei") or "").strip()
            if not lei:
                continue
            companies[lei] = {
                "name": row["company_name"],
                "cni_sector": row["cni_sector"],
                "isic_code": row["isic_sector_code"],
                "isic_name": row["isic_sector_name"],
                "source_type": row["source_type"],
            }
    print(f"Loaded {len(companies)} companies from manifest.")

    # 2) Group FR metadata rows by LEI for target companies.
    rows_by_lei: dict[str, list[dict]] = defaultdict(list)
    with fr_metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lei = (row.get("company__lei") or "").strip()
            if lei in companies:
                rows_by_lei[lei].append(row)

    # 3) Select one filing per company-year.
    queue = []
    missing_log = []
    target_years_set = set(target_years)

    for lei, info in companies.items():
        raw_rows = rows_by_lei.get(lei, [])
        candidates: list[Candidate] = []
        for row in raw_rows:
            candidate = _build_candidate(row, target_years_set)
            if candidate is not None:
                candidates.append(candidate)

        selected = _select_best_by_year(candidates, target_years)

        for year in target_years:
            chosen = selected.get(year)
            if chosen is None:
                missing_log.append(f"{info['name']} ({year}): No candidate filing found")
                continue

            md_path = fr_markdown_dir / f"{chosen.pk}.md"
            if not md_path.exists():
                missing_log.append(
                    f"{info['name']} ({year}): Markdown file missing for PK {chosen.pk}"
                )
                continue

            queue.append(
                {
                    "company_name": info["name"],
                    "lei": lei,
                    "year": year,
                    "cni_sector": info["cni_sector"],
                    "isic_sector": info["isic_name"],
                    "file_path": str(md_path),
                    "pk": chosen.pk,
                    "source_title": chosen.title,
                }
            )

    # Stable output ordering.
    queue.sort(key=lambda x: (x["company_name"], x["year"]))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)

    print(f"\nGenerated queue with {len(queue)} items.")
    print(f"Saved to {output_file}")

    if missing_log:
        print(f"\nMissing Reports ({len(missing_log)}):")
        for msg in missing_log[:20]:
            print(f"  - {msg}")
        if len(missing_log) > 20:
            print(f"  ... and {len(missing_log) - 20} more.")


def main():
    parser = argparse.ArgumentParser(description="Generate FR processing queue for company-year targets.")
    parser.add_argument(
        "--years",
        default="2023,2024",
        help="Comma-separated years to include, e.g. 2022,2023,2024",
    )
    parser.add_argument(
        "--output",
        default=str(output_path),
        help="Output JSON path for the generated queue.",
    )
    args = parser.parse_args()

    target_years = _parse_years_csv(args.years)
    generate_queue(target_years=target_years, output_file=Path(args.output))


if __name__ == "__main__":
    main()
