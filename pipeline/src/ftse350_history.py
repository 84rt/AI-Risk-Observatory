"""Build year-end FTSE 350 membership snapshots from archived constituent pages.

The collector prefers official London Stock Exchange constituent pages archived
via the Internet Archive Wayback API. It can optionally reconcile against
official quarterly review notices supplied in JSON form.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from html import unescape
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import requests

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a declared dependency
    pd = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "reference" / "ftse350_history"
WAYBACK_AVAILABILITY_API = "https://archive.org/wayback/available"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_TIMEOUT_SEC = 45
USER_AGENT = "AIRiskObservatory/1.0 (+https://github.com/openai)"

SEGMENT_EXPECTED_COUNTS = {
    "FTSE 100": 100,
    "FTSE 250": 250,
}

OFFICIAL_CONSTITUENT_URLS = {
    "FTSE 100": [
        "https://www.londonstockexchange.com/indices/ftse-100/constituents/table",
    ],
    "FTSE 250": [
        "https://www.londonstockexchange.com/indices/ftse-250/constituents/table",
    ],
}

WIKIPEDIA_PAGE_TITLES = {
    "FTSE 100": "FTSE 100 Index",
    "FTSE 250": "FTSE 250 Index",
}

WIKIPEDIA_PAGE_URLS = {
    "FTSE 100": "https://en.wikipedia.org/wiki/FTSE_100_Index",
    "FTSE 250": "https://en.wikipedia.org/wiki/FTSE_250_Index",
}

NAME_HEADER_KEYWORDS = (
    "company",
    "constituent",
    "security",
    "issuer",
    "name",
)
TICKER_HEADER_KEYWORDS = (
    "ticker",
    "epic",
    "symbol",
    "mnemonic",
    "ric",
)

class FTSEHistoryError(RuntimeError):
    """Base error for FTSE history collection."""


class SnapshotNotFoundError(FTSEHistoryError):
    """Raised when no acceptable archived official snapshot can be found."""


class ValidationError(FTSEHistoryError):
    """Raised when collected membership fails hard validation."""


@dataclass(frozen=True)
class YearReviewInfo:
    year: int
    effective_date: str | None
    review_notice_urls: list[str]
    notices_supplied: bool


@dataclass(frozen=True)
class ArchivedSnapshot:
    year: int
    segment: str
    source_url: str
    archive_url: str
    snapshot_date: str
    approximate_snapshot: bool
    source_type: str
    html_path: str | None = None


@dataclass(frozen=True)
class ConstituentRow:
    year: int
    segment: str
    company_name: str
    company_name_raw: str
    normalized_company_name: str
    ticker_epic: str
    source_type: str
    source_url: str
    archive_url: str
    snapshot_date: str
    effective_date: str
    approximate_snapshot: bool
    validation_status: str
    validation_notes: str


@dataclass(frozen=True)
class ReviewNotice:
    url: str
    effective_date: str | None
    additions: dict[str, list[str]]
    deletions: dict[str, list[str]]


class _HTMLTableParser(HTMLParser):
    """Minimal HTML table extractor used when pandas is insufficient."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._current_cell = []
        elif self._in_cell and tag == "br":
            self._current_cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._in_table:
            if self._current_table:
                self.tables.append(self._current_table)
            self._in_table = False
        elif tag == "tr" and self._in_row:
            if any(cell.strip() for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._in_row = False
        elif tag in {"td", "th"} and self._in_cell:
            text = clean_cell_text("".join(self._current_cell))
            self._current_row.append(text)
            self._in_cell = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Calendar years to build year-end FTSE 350 snapshots for.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--refresh-web",
        action="store_true",
        help="Force refetching archive availability and snapshot HTML.",
    )
    parser.add_argument(
        "--max-snapshot-gap-days",
        type=int,
        default=45,
        help="Maximum allowed distance between year-end and archived capture.",
    )
    parser.add_argument(
        "--validate-deltas",
        action="store_true",
        help="Validate collected snapshots against supplied quarterly review notices.",
    )
    parser.add_argument(
        "--write-raw",
        action="store_true",
        help="Persist fetched raw HTML and metadata beneath the output directory.",
    )
    parser.add_argument(
        "--review-notices-json",
        default=None,
        help="Optional JSON file describing official review notices and effective dates.",
    )
    return parser.parse_args(argv)


def clean_cell_text(value: str) -> str:
    text = unescape(value or "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_company_name(name: str) -> str:
    text = clean_cell_text(name)
    text = text.translate(
        str.maketrans(
            {
                "’": "'",
                "“": '"',
                "”": '"',
                "–": "-",
                "—": "-",
            }
        )
    )
    text = re.sub(r"\s+\([^)]*\)$", "", text)
    text = re.sub(r"\bordinary shares?\b.*$", "", text, flags=re.I)
    text = re.sub(r"\b(ord|ords)\b.*$", "", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9&+'/-]+", " ", text)
    text = text.lower()
    text = re.sub(r"\b(public limited company|plc)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_ticker(value: str | None) -> str:
    if not value:
        return ""
    text = clean_cell_text(value).upper()
    text = re.sub(r"^(LON|LSE)\s*:\s*", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^A-Z0-9.\-]", "", text)
    return text


def year_end_date(year: int) -> date:
    return date(year, 12, 31)


def iso_from_wayback_timestamp(timestamp: str) -> str:
    return datetime.strptime(timestamp, "%Y%m%d%H%M%S").date().isoformat()


def slugify_segment(segment: str) -> str:
    return segment.lower().replace(" ", "_")


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        }
    )
    return session


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_review_info(path: str | None) -> dict[int, YearReviewInfo]:
    if not path:
        return {}

    payload = read_json(Path(path))
    review_map: dict[int, YearReviewInfo] = {}
    for key, value in payload.items():
        year = int(key)
        if not isinstance(value, dict):
            raise FTSEHistoryError(f"Review notice config for {year} must be an object.")
        urls = value.get("review_notice_urls") or value.get("notices") or []
        if not isinstance(urls, list):
            raise FTSEHistoryError(f"Review notice URLs for {year} must be a list.")
        review_map[year] = YearReviewInfo(
            year=year,
            effective_date=value.get("effective_date"),
            review_notice_urls=[str(item) for item in urls],
            notices_supplied=bool(urls),
        )
    return review_map


def cache_paths(
    out_dir: Path,
    year: int,
    segment: str,
    *,
    source_name: str = "official",
) -> tuple[Path, Path]:
    raw_dir = out_dir / "raw" / str(year)
    stem = slugify_segment(segment)
    suffix = "" if source_name == "official" else f".{source_name}"
    return raw_dir / f"{stem}{suffix}.html", raw_dir / f"{stem}{suffix}.meta.json"


def fetch_wayback_snapshot(
    session: requests.Session,
    source_url: str,
    target: date,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any] | None:
    params = {
        "url": source_url,
        "timestamp": target.strftime("%Y%m%d235959"),
    }
    response = session.get(WAYBACK_AVAILABILITY_API, params=params, timeout=timeout_sec)
    response.raise_for_status()
    payload = response.json()
    closest = payload.get("archived_snapshots", {}).get("closest")
    if not closest or not closest.get("available"):
        return None
    return closest


def resolve_archived_snapshot(
    session: requests.Session,
    out_dir: Path,
    year: int,
    segment: str,
    max_snapshot_gap_days: int,
    refresh_web: bool,
    write_raw: bool,
) -> tuple[ArchivedSnapshot, str]:
    html_cache_path, meta_cache_path = cache_paths(out_dir, year, segment, source_name="official")
    if not refresh_web and html_cache_path.exists() and meta_cache_path.exists():
        cached_meta = read_json(meta_cache_path)
        snapshot = ArchivedSnapshot(
            year=year,
            segment=segment,
            source_url=cached_meta["source_url"],
            archive_url=cached_meta["archive_url"],
            snapshot_date=cached_meta["snapshot_date"],
            approximate_snapshot=bool(cached_meta.get("approximate_snapshot")),
            source_type=cached_meta["source_type"],
            html_path=str(html_cache_path),
        )
        return snapshot, html_cache_path.read_text(encoding="utf-8")

    target = year_end_date(year)
    last_error: SnapshotNotFoundError | None = None

    for source_url in OFFICIAL_CONSTITUENT_URLS[segment]:
        closest = fetch_wayback_snapshot(session, source_url=source_url, target=target)
        if not closest:
            last_error = SnapshotNotFoundError(
                f"No Wayback snapshot found for {segment} {year} at {source_url}."
            )
            continue

        snapshot_date = iso_from_wayback_timestamp(closest["timestamp"])
        gap_days = abs((date.fromisoformat(snapshot_date) - target).days)
        if gap_days > max_snapshot_gap_days:
            last_error = SnapshotNotFoundError(
                f"Closest snapshot for {segment} {year} is {gap_days} days away "
                f"({snapshot_date}) from year-end."
            )
            continue

        archive_url = closest["url"]
        response = session.get(archive_url, timeout=DEFAULT_TIMEOUT_SEC)
        response.raise_for_status()
        html = response.text
        source_type = "official_archived" if gap_days == 0 else "official_archived_nearby"
        snapshot = ArchivedSnapshot(
            year=year,
            segment=segment,
            source_url=source_url,
            archive_url=archive_url,
            snapshot_date=snapshot_date,
            approximate_snapshot=gap_days != 0,
            source_type=source_type,
            html_path=str(html_cache_path) if write_raw else None,
        )

        if write_raw:
            html_cache_path.parent.mkdir(parents=True, exist_ok=True)
            html_cache_path.write_text(html, encoding="utf-8")
            write_json(meta_cache_path, asdict(snapshot))

        return snapshot, html

    if last_error:
        raise last_error
    raise SnapshotNotFoundError(f"Unable to resolve archived snapshot for {segment} {year}.")


def fetch_wikipedia_revision_candidates(
    session: requests.Session,
    *,
    title: str,
    start_timestamp: str,
    direction: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    response = session.get(
        WIKIPEDIA_API_URL,
        params={
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "titles": title,
            "rvlimit": limit,
            "rvdir": direction,
            "rvstart": start_timestamp,
            "rvprop": "ids|timestamp",
        },
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        return []
    page = next(iter(pages.values()))
    revisions = page.get("revisions") or []
    return [revision for revision in revisions if revision.get("revid") and revision.get("timestamp")]


def fetch_wikipedia_revision_html(
    session: requests.Session,
    *,
    oldid: int,
) -> str:
    response = session.get(
        WIKIPEDIA_API_URL,
        params={
            "action": "parse",
            "format": "json",
            "oldid": oldid,
            "prop": "text",
        },
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["parse"]["text"]["*"]


def resolve_wikipedia_fallback_snapshot(
    session: requests.Session,
    out_dir: Path,
    *,
    year: int,
    segment: str,
    effective_date: str,
    max_snapshot_gap_days: int,
    refresh_web: bool,
    write_raw: bool,
) -> tuple[ArchivedSnapshot, list[ConstituentRow]]:
    html_cache_path, meta_cache_path = cache_paths(out_dir, year, segment, source_name="wikipedia")
    if not refresh_web and html_cache_path.exists() and meta_cache_path.exists():
        cached_meta = read_json(meta_cache_path)
        snapshot = ArchivedSnapshot(
            year=year,
            segment=segment,
            source_url=cached_meta["source_url"],
            archive_url=cached_meta["archive_url"],
            snapshot_date=cached_meta["snapshot_date"],
            approximate_snapshot=bool(cached_meta.get("approximate_snapshot")),
            source_type=cached_meta["source_type"],
            html_path=str(html_cache_path),
        )
        rows = parse_constituents_html(
            html_cache_path.read_text(encoding="utf-8"),
            year=year,
            segment=segment,
            source_type=snapshot.source_type,
            source_url=snapshot.source_url,
            archive_url=snapshot.archive_url,
            snapshot_date=snapshot.snapshot_date,
            effective_date=effective_date,
            approximate_snapshot=snapshot.approximate_snapshot,
        )
        return snapshot, rows

    target = year_end_date(year)
    target_after_year_end = f"{year + 1}-01-01T00:00:00Z"
    target_at_year_end = f"{year}-12-31T23:59:59Z"
    title = WIKIPEDIA_PAGE_TITLES[segment]
    expected = SEGMENT_EXPECTED_COUNTS[segment]
    candidates: list[tuple[int, int, str, int]] = []

    for direction, start_timestamp in (
        ("older", target_after_year_end),
        ("newer", target_at_year_end),
    ):
        revisions = fetch_wikipedia_revision_candidates(
            session,
            title=title,
            start_timestamp=start_timestamp,
            direction=direction,
        )
        for revision in revisions:
            snapshot_date = revision["timestamp"][:10]
            gap_days = abs((date.fromisoformat(snapshot_date) - target).days)
            if gap_days > max_snapshot_gap_days:
                continue
            candidates.append((gap_days, 1 if direction == "newer" else 0, snapshot_date, int(revision["revid"])))

    for gap_days, _, snapshot_date, oldid in sorted(set(candidates)):
        archive_url = f"https://en.wikipedia.org/w/index.php?oldid={oldid}"
        html = fetch_wikipedia_revision_html(session, oldid=oldid)
        snapshot = ArchivedSnapshot(
            year=year,
            segment=segment,
            source_url=WIKIPEDIA_PAGE_URLS[segment],
            archive_url=archive_url,
            snapshot_date=snapshot_date,
            approximate_snapshot=gap_days != 0,
            source_type="fallback_secondary",
            html_path=str(html_cache_path) if write_raw else None,
        )
        try:
            rows = parse_constituents_html(
                html,
                year=year,
                segment=segment,
                source_type=snapshot.source_type,
                source_url=snapshot.source_url,
                archive_url=snapshot.archive_url,
                snapshot_date=snapshot.snapshot_date,
                effective_date=effective_date,
                approximate_snapshot=snapshot.approximate_snapshot,
            )
        except FTSEHistoryError:
            continue

        if len(rows) != expected:
            continue

        if write_raw:
            html_cache_path.parent.mkdir(parents=True, exist_ok=True)
            html_cache_path.write_text(html, encoding="utf-8")
            write_json(meta_cache_path, asdict(snapshot))

        return snapshot, rows

    if not candidates:
        raise SnapshotNotFoundError(
            f"No valid Wikipedia fallback snapshot found for {segment} {year} within "
            f"{max_snapshot_gap_days} days of year-end."
        )
    raise SnapshotNotFoundError(
        f"Wikipedia revisions were found for {segment} {year}, but none parsed to the "
        f"expected {expected} constituents within {max_snapshot_gap_days} days of year-end."
    )


def extract_tables_with_pandas(html: str) -> list[list[list[str]]]:
    if pd is None:
        return []
    try:
        frames = pd.read_html(StringIO(html))
    except ValueError:
        return []
    tables: list[list[list[str]]] = []
    for frame in frames:
        table: list[list[str]] = [list(frame.columns.astype(str))]
        for _, row in frame.iterrows():
            table.append([clean_cell_text(str(value)) for value in row.tolist()])
        tables.append(table)
    return tables


def extract_tables_from_html(html: str) -> list[list[list[str]]]:
    parser = _HTMLTableParser()
    parser.feed(html)
    tables = parser.tables
    if tables:
        return tables
    return extract_tables_with_pandas(html)


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def is_header_row(row: list[str]) -> bool:
    normalized = [normalize_header(cell) for cell in row]
    if not normalized:
        return False
    joined = " ".join(normalized)
    if any(keyword in joined for keyword in NAME_HEADER_KEYWORDS + TICKER_HEADER_KEYWORDS):
        return True
    alpha_cells = [cell for cell in normalized if re.search(r"[a-z]", cell)]
    if not alpha_cells:
        return False
    return all(len(cell) <= 24 for cell in alpha_cells) and len(alpha_cells) == len(normalized)


def table_to_records(table: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    rows = [[clean_cell_text(cell) for cell in row] for row in table if any(clean_cell_text(cell) for cell in row)]
    if not rows:
        return [], []

    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    if is_header_row(padded_rows[0]):
        headers = padded_rows[0]
        body = padded_rows[1:]
    else:
        headers = [f"column_{idx + 1}" for idx in range(width)]
        body = padded_rows
    return headers, body


def find_name_column(headers: list[str], rows: list[list[str]]) -> int | None:
    normalized_headers = [normalize_header(header) for header in headers]
    for idx, header in enumerate(normalized_headers):
        if any(keyword in header for keyword in NAME_HEADER_KEYWORDS):
            return idx

    if len(headers) >= 2 and rows:
        if sum(1 for row in rows if row and row[0].isdigit()) >= max(3, len(rows) // 3):
            return 1
    if headers:
        return 0
    return None


def find_ticker_column(headers: list[str]) -> int | None:
    normalized_headers = [normalize_header(header) for header in headers]
    for idx, header in enumerate(normalized_headers):
        if any(keyword in header for keyword in TICKER_HEADER_KEYWORDS):
            return idx
    return None


def score_table(table: list[list[str]], segment: str) -> tuple[int, list[str], list[list[str]], int | None, int | None]:
    headers, rows = table_to_records(table)
    if not rows:
        return -1, headers, rows, None, None

    name_col = find_name_column(headers, rows)
    ticker_col = find_ticker_column(headers)
    score = 0

    expected = SEGMENT_EXPECTED_COUNTS[segment]
    row_count = len(rows)
    if row_count >= max(10, expected // 3):
        score += 10
    if abs(row_count - expected) <= expected:
        score += 15
    if name_col is not None:
        score += 40
    if ticker_col is not None:
        score += 10
    if rows and name_col is not None:
        non_empty_names = sum(1 for row in rows if row[name_col].strip())
        score += min(non_empty_names, 30)

    return score, headers, rows, name_col, ticker_col


def parse_constituents_html(
    html: str,
    *,
    year: int,
    segment: str,
    source_type: str,
    source_url: str,
    archive_url: str,
    snapshot_date: str,
    effective_date: str,
    approximate_snapshot: bool,
) -> list[ConstituentRow]:
    tables = extract_tables_from_html(html)
    scored_tables = [
        score_table(table, segment)
        for table in tables
    ]
    if not scored_tables:
        raise FTSEHistoryError(f"No HTML tables found for {segment} {year}.")

    best_score, headers, rows, name_col, ticker_col = max(scored_tables, key=lambda item: item[0])
    if best_score < 35 or name_col is None:
        raise FTSEHistoryError(
            f"Could not identify a constituent table for {segment} {year}; best score={best_score}."
        )

    records: list[ConstituentRow] = []
    for row in rows:
        raw_name = clean_cell_text(row[name_col])
        if not raw_name:
            continue
        normalized_name = normalize_company_name(raw_name)
        if not normalized_name:
            continue
        ticker = normalize_ticker(row[ticker_col]) if ticker_col is not None and ticker_col < len(row) else ""
        display_name = clean_cell_text(re.sub(r"\s+\([^)]*\)$", "", raw_name))
        records.append(
            ConstituentRow(
                year=year,
                segment=segment,
                company_name=display_name,
                company_name_raw=raw_name,
                normalized_company_name=normalized_name,
                ticker_epic=ticker,
                source_type=source_type,
                source_url=source_url,
                archive_url=archive_url,
                snapshot_date=snapshot_date,
                effective_date=effective_date,
                approximate_snapshot=approximate_snapshot,
                validation_status="pending",
                validation_notes="",
            )
        )
    return dedupe_constituents(records)


def dedupe_constituents(rows: Iterable[ConstituentRow]) -> list[ConstituentRow]:
    deduped: dict[str, ConstituentRow] = {}
    for row in rows:
        key = row.normalized_company_name
        if row.ticker_epic:
            key = f"{key}|{row.ticker_epic}"
        deduped.setdefault(key, row)
    return list(deduped.values())


def load_review_notices(session: requests.Session, review_info: YearReviewInfo | None) -> list[ReviewNotice]:
    if not review_info or not review_info.review_notice_urls:
        return []

    notices: list[ReviewNotice] = []
    for url in review_info.review_notice_urls:
        response = session.get(url, timeout=DEFAULT_TIMEOUT_SEC)
        response.raise_for_status()
        notices.append(parse_review_notice(url, response.text))
    return notices


def parse_review_notice(url: str, text: str) -> ReviewNotice:
    cleaned = clean_cell_text(re.sub(r"<[^>]+>", " ", text))
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    date_match = re.search(
        r"(effective(?:\s+from)?|changes?\s+take\s+effect(?:\s+from)?|after\s+the\s+close\s+of\s+business\s+on)\s+"
        r"(\d{1,2}\s+[A-Za-z]+\s+20\d{2})",
        cleaned,
        flags=re.I,
    )
    effective_date = None
    if date_match:
        effective_date = datetime.strptime(date_match.group(2), "%d %B %Y").date().isoformat()

    additions = {"FTSE 100": [], "FTSE 250": []}
    deletions = {"FTSE 100": [], "FTSE 250": []}

    sentence_patterns = (
        ("FTSE 100", additions, re.compile(r"([A-Z][A-Za-z0-9&'.,\- ]{2,})\s+will be added to the FTSE 100 Index", re.I)),
        ("FTSE 100", deletions, re.compile(r"([A-Z][A-Za-z0-9&'.,\- ]{2,})\s+will be deleted from the FTSE 100 Index", re.I)),
        ("FTSE 250", additions, re.compile(r"([A-Z][A-Za-z0-9&'.,\- ]{2,})\s+will be added to the FTSE 250 Index", re.I)),
        ("FTSE 250", deletions, re.compile(r"([A-Z][A-Za-z0-9&'.,\- ]{2,})\s+will be deleted from the FTSE 250 Index", re.I)),
    )
    for segment, bucket, pattern in sentence_patterns:
        for sentence in sentences:
            for match in pattern.findall(sentence):
                bucket[segment].append(clean_cell_text(match))

    return ReviewNotice(
        url=url,
        effective_date=effective_date,
        additions=additions,
        deletions=deletions,
    )


def validate_rows(
    rows: list[ConstituentRow],
    *,
    year: int,
    review_info: YearReviewInfo | None,
    review_notices: list[ReviewNotice],
) -> tuple[list[ConstituentRow], dict[str, Any], bool]:
    grouped: dict[str, list[ConstituentRow]] = {
        segment: [row for row in rows if row.segment == segment]
        for segment in SEGMENT_EXPECTED_COUNTS
    }
    validation_summary: dict[str, Any] = {
        "year": year,
        "segment_counts": {},
        "total_count": len(rows),
        "duplicate_keys": [],
        "warnings": [],
        "review_notice_urls": [notice.url for notice in review_notices],
        "effective_date": review_info.effective_date if review_info else None,
    }

    failed = False
    for segment, expected in SEGMENT_EXPECTED_COUNTS.items():
        actual = len(grouped.get(segment, []))
        segment_ok = actual == expected
        validation_summary["segment_counts"][segment] = {
            "expected": expected,
            "actual": actual,
            "status": "pass" if segment_ok else "failed",
        }
        if not segment_ok:
            failed = True
            validation_summary["warnings"].append(
                f"{segment} row count was {actual}; expected {expected}."
            )

    total_ok = len(rows) == 350
    validation_summary["total_status"] = "pass" if total_ok else "failed"
    if not total_ok:
        failed = True
        validation_summary["warnings"].append(
            f"Combined FTSE 350 row count was {len(rows)}; expected 350."
        )

    seen: dict[str, int] = {}
    for row in rows:
        key = row.normalized_company_name if not row.ticker_epic else f"{row.normalized_company_name}|{row.ticker_epic}"
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    validation_summary["duplicate_keys"] = duplicates
    if duplicates:
        failed = True
        validation_summary["warnings"].append(
            f"Found {len(duplicates)} duplicate constituent identity keys."
        )

    effective_date = None
    if review_notices:
        effective_dates = sorted({notice.effective_date for notice in review_notices if notice.effective_date})
        if len(effective_dates) == 1:
            effective_date = effective_dates[0]
        elif len(effective_dates) > 1:
            validation_summary["warnings"].append(
                f"Review notices disagree on effective date: {', '.join(effective_dates)}."
            )
            failed = True
    if not effective_date and review_info and review_info.effective_date:
        effective_date = review_info.effective_date
    if not effective_date and rows:
        effective_date = rows[0].snapshot_date
        validation_summary["warnings"].append(
            "No official effective date was supplied; using snapshot date as effective date."
        )

    validated_rows: list[ConstituentRow] = []
    status = "failed" if failed else ("warning" if validation_summary["warnings"] else "pass")
    notes = "; ".join(validation_summary["warnings"])
    for row in rows:
        validated_rows.append(
            ConstituentRow(
                **{
                    **asdict(row),
                    "effective_date": effective_date or row.effective_date,
                    "validation_status": status,
                    "validation_notes": notes,
                }
            )
        )
    validation_summary["resolved_effective_date"] = effective_date
    validation_summary["status"] = status
    return validated_rows, validation_summary, failed


def collect_segment_rows(
    session: requests.Session,
    *,
    year: int,
    segment: str,
    out_dir: Path,
    effective_date: str,
    max_snapshot_gap_days: int,
    refresh_web: bool,
    write_raw: bool,
) -> tuple[list[ConstituentRow], dict[str, Any]]:
    errors: list[str] = []

    try:
        snapshot, html = resolve_archived_snapshot(
            session,
            out_dir=out_dir,
            year=year,
            segment=segment,
            max_snapshot_gap_days=max_snapshot_gap_days,
            refresh_web=refresh_web,
            write_raw=write_raw,
        )
        segment_rows = parse_constituents_html(
            html,
            year=year,
            segment=segment,
            source_type=snapshot.source_type,
            source_url=snapshot.source_url,
            archive_url=snapshot.archive_url,
            snapshot_date=snapshot.snapshot_date,
            effective_date=effective_date,
            approximate_snapshot=snapshot.approximate_snapshot,
        )
        if len(segment_rows) == SEGMENT_EXPECTED_COUNTS[segment]:
            return segment_rows, {
                **asdict(snapshot),
                "row_count": len(segment_rows),
                "expected_count": SEGMENT_EXPECTED_COUNTS[segment],
                "collection_notes": [],
            }
        errors.append(
            f"Official archived parse produced {len(segment_rows)} rows; "
            f"expected {SEGMENT_EXPECTED_COUNTS[segment]}."
        )
    except FTSEHistoryError as exc:
        errors.append(str(exc))

    fallback_snapshot, fallback_rows = resolve_wikipedia_fallback_snapshot(
        session,
        out_dir=out_dir,
        year=year,
        segment=segment,
        effective_date=effective_date,
        max_snapshot_gap_days=max_snapshot_gap_days,
        refresh_web=refresh_web,
        write_raw=write_raw,
    )
    return fallback_rows, {
        **asdict(fallback_snapshot),
        "row_count": len(fallback_rows),
        "expected_count": SEGMENT_EXPECTED_COUNTS[segment],
        "collection_notes": errors,
    }


def build_year_snapshot(
    session: requests.Session,
    *,
    year: int,
    out_dir: Path,
    max_snapshot_gap_days: int,
    refresh_web: bool,
    validate_deltas: bool,
    write_raw: bool,
    review_info: YearReviewInfo | None,
) -> tuple[list[ConstituentRow], dict[str, Any], dict[str, Any], bool]:
    rows: list[ConstituentRow] = []
    provenance_segments: dict[str, Any] = {}
    effective_date = review_info.effective_date if review_info and review_info.effective_date else year_end_date(year).isoformat()

    for segment in SEGMENT_EXPECTED_COUNTS:
        segment_rows, segment_provenance = collect_segment_rows(
            session,
            year=year,
            segment=segment,
            out_dir=out_dir,
            effective_date=effective_date,
            max_snapshot_gap_days=max_snapshot_gap_days,
            refresh_web=refresh_web,
            write_raw=write_raw,
        )
        rows.extend(segment_rows)
        provenance_segments[segment] = segment_provenance

    review_notices = load_review_notices(session, review_info) if validate_deltas else []
    validated_rows, validation_summary, failed = validate_rows(
        rows,
        year=year,
        review_info=review_info,
        review_notices=review_notices,
    )

    provenance = {
        "year": year,
        "segments": provenance_segments,
        "review_notice_urls": review_info.review_notice_urls if review_info else [],
        "review_notices_supplied": bool(review_info and review_info.notices_supplied),
        "effective_date": validation_summary.get("resolved_effective_date"),
    }
    return validated_rows, provenance, validation_summary, failed


def rows_to_dicts(rows: Iterable[ConstituentRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def write_rows_csv(path: Path, rows: Iterable[ConstituentRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = rows_to_dicts(rows)
    if not rows_list:
        raise FTSEHistoryError(f"No rows to write for {path}.")

    fieldnames = [
        "year",
        "segment",
        "company_name",
        "company_name_raw",
        "normalized_company_name",
        "ticker_epic",
        "source_type",
        "source_url",
        "archive_url",
        "snapshot_date",
        "effective_date",
        "approximate_snapshot",
        "validation_status",
        "validation_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def build_history(
    *,
    years: list[int],
    out_dir: Path,
    refresh_web: bool,
    max_snapshot_gap_days: int,
    validate_deltas: bool,
    write_raw: bool,
    review_notices_json: str | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = build_session()
    review_map = load_review_info(review_notices_json)

    combined_rows: list[ConstituentRow] = []
    provenance: dict[str, Any] = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "years": {},
        "assumptions": [
            "Annual snapshots target year-end membership.",
            "Official LSE constituent pages archived in Wayback are preferred.",
            "If official review notices are not supplied, snapshot date is used as effective_date.",
        ],
    }
    validation_report: dict[str, Any] = {
        "generated_at": provenance["generated_at"],
        "years": {},
    }

    hard_failures: list[str] = []
    for year in years:
        rows, year_provenance, year_validation, failed = build_year_snapshot(
            session,
            year=year,
            out_dir=out_dir,
            max_snapshot_gap_days=max_snapshot_gap_days,
            refresh_web=refresh_web,
            validate_deltas=validate_deltas,
            write_raw=write_raw,
            review_info=review_map.get(year),
        )
        combined_rows.extend(rows)
        provenance["years"][str(year)] = year_provenance
        validation_report["years"][str(year)] = year_validation
        write_rows_csv(out_dir / f"ftse350_{year}.csv", rows)
        if failed:
            hard_failures.append(f"{year}: {year_validation['status']}")

    combined_rows.sort(key=lambda row: (row.year, row.segment, row.company_name.lower()))
    write_rows_csv(out_dir / f"ftse350_{years[0]}_{years[-1]}_long.csv", combined_rows)
    write_json(out_dir / "provenance.json", provenance)
    write_json(out_dir / "validation_report.json", validation_report)

    if hard_failures:
        raise ValidationError(
            "One or more yearly snapshots failed validation: " + ", ".join(hard_failures)
        )

    return {
        "rows_written": len(combined_rows),
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_history(
            years=sorted(set(args.years)),
            out_dir=Path(args.out_dir),
            refresh_web=args.refresh_web,
            max_snapshot_gap_days=args.max_snapshot_gap_days,
            validate_deltas=args.validate_deltas,
            write_raw=args.write_raw,
            review_notices_json=args.review_notices_json,
        )
    except FTSEHistoryError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Wrote FTSE 350 history to {result['out_dir']} ({result['rows_written']} rows total).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
