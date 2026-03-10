from __future__ import annotations

from pathlib import Path

import pytest

from src.ftse350_history import (
    SnapshotNotFoundError,
    ValidationError,
    build_history,
    normalize_company_name,
    normalize_ticker,
    parse_constituents_html,
    parse_review_notice,
    validate_rows,
)


FTSE100_HTML = """
<html>
  <body>
    <table>
      <thead>
        <tr><th>Company</th><th>EPIC</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </body>
</html>
"""

FTSE250_HTML = """
<html>
  <body>
    <table>
      <thead>
        <tr><th>Issuer name</th><th>Ticker</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </body>
</html>
"""


def make_rows(prefix: str, count: int) -> str:
    return "\n".join(
        f"<tr><td>{prefix} Company {idx} plc</td><td>{prefix}{idx:03d}</td></tr>"
        for idx in range(1, count + 1)
    )


def test_parse_constituents_html_extracts_names_and_tickers() -> None:
    html = FTSE100_HTML.format(rows=make_rows("AAA", 100))
    rows = parse_constituents_html(
        html,
        year=2025,
        segment="FTSE 100",
        source_type="official_archived",
        source_url="https://example.com/ftse100",
        archive_url="https://web.archive.org/example/ftse100",
        snapshot_date="2025-12-31",
        effective_date="2025-12-31",
        approximate_snapshot=False,
    )

    assert len(rows) == 100
    assert rows[0].company_name == "AAA Company 1 plc"
    assert rows[0].ticker_epic == "AAA001"
    assert rows[0].normalized_company_name == "aaa company 1"


def test_normalization_strips_suffix_noise() -> None:
    assert normalize_company_name("Marks & Spencer Group plc") == "marks & spencer group"
    assert normalize_company_name("Whitbread PLC Ordinary Shares") == "whitbread"
    assert normalize_ticker("LON: cna ") == "CNA"


def test_parse_review_notice_extracts_effective_date() -> None:
    html = """
    <html><body>
      <p>Changes take effect from 20 December 2024.</p>
      <p>Alpha plc will be added to the FTSE 100 Index.</p>
      <p>Beta plc will be deleted from the FTSE 250 Index.</p>
    </body></html>
    """
    notice = parse_review_notice("https://example.com/notice", html)
    assert notice.effective_date == "2024-12-20"
    assert "Alpha plc" in notice.additions["FTSE 100"]
    assert "Beta plc" in notice.deletions["FTSE 250"]


def test_validate_rows_flags_wrong_count() -> None:
    html = FTSE100_HTML.format(rows=make_rows("AAA", 99))
    rows = parse_constituents_html(
        html,
        year=2025,
        segment="FTSE 100",
        source_type="official_archived",
        source_url="https://example.com/ftse100",
        archive_url="https://web.archive.org/example/ftse100",
        snapshot_date="2025-12-31",
        effective_date="2025-12-31",
        approximate_snapshot=False,
    )
    validated, summary, failed = validate_rows(rows, year=2025, review_info=None, review_notices=[])
    assert failed is True
    assert summary["segment_counts"]["FTSE 100"]["actual"] == 99
    assert validated[0].validation_status == "failed"


def test_build_history_writes_expected_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        def __init__(self, *, json_payload=None, text="", status_code=200):
            self._json_payload = json_payload
            self.text = text
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._json_payload

    class DummySession:
        def get(self, url, params=None, timeout=None):
            if "wayback/available" in url:
                target = params["url"]
                segment = "FTSE 100" if "ftse100" in target else "FTSE 250"
                timestamp = "20251231120000"
                return DummyResponse(
                    json_payload={
                        "archived_snapshots": {
                            "closest": {
                                "available": True,
                                "timestamp": timestamp,
                                "url": f"https://web.archive.org/{segment.replace(' ', '').lower()}",
                            }
                        }
                    }
                )
            if "ftse100" in url:
                return DummyResponse(text=FTSE100_HTML.format(rows=make_rows("AAA", 100)))
            if "ftse250" in url:
                return DummyResponse(text=FTSE250_HTML.format(rows=make_rows("BBB", 250)))
            if "ftse100" not in url and "ftse250" not in url and "web.archive.org" in url:
                if "ftse100" in url.lower():
                    return DummyResponse(text=FTSE100_HTML.format(rows=make_rows("AAA", 100)))
                return DummyResponse(text=FTSE250_HTML.format(rows=make_rows("BBB", 250)))
            raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr("src.ftse350_history.build_session", lambda: DummySession())
    monkeypatch.setattr(
        "src.ftse350_history.OFFICIAL_CONSTITUENT_URLS",
        {
            "FTSE 100": ["https://example.com/ftse100"],
            "FTSE 250": ["https://example.com/ftse250"],
        },
    )

    result = build_history(
        years=[2025],
        out_dir=tmp_path,
        refresh_web=True,
        max_snapshot_gap_days=45,
        validate_deltas=False,
        write_raw=False,
        review_notices_json=None,
    )

    assert result["rows_written"] == 350
    assert (tmp_path / "ftse350_2025.csv").exists()
    assert (tmp_path / "ftse350_2025_2025_long.csv").exists()
    assert (tmp_path / "provenance.json").exists()
    assert (tmp_path / "validation_report.json").exists()


def test_build_history_raises_on_validation_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        def __init__(self, *, json_payload=None, text="", status_code=200):
            self._json_payload = json_payload
            self.text = text
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._json_payload

    class DummySession:
        def get(self, url, params=None, timeout=None):
            if "wayback/available" in url:
                target = params["url"]
                segment = "ftse100" if "ftse100" in target else "ftse250"
                return DummyResponse(
                    json_payload={
                        "archived_snapshots": {
                            "closest": {
                                "available": True,
                                "timestamp": "20251231120000",
                                "url": f"https://web.archive.org/{segment}",
                            }
                        }
                    }
                )
            if "ftse100" in url:
                return DummyResponse(text=FTSE100_HTML.format(rows=make_rows("AAA", 99)))
            return DummyResponse(text=FTSE250_HTML.format(rows=make_rows("BBB", 250)))

    monkeypatch.setattr("src.ftse350_history.build_session", lambda: DummySession())
    monkeypatch.setattr(
        "src.ftse350_history.resolve_wikipedia_fallback_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SnapshotNotFoundError("No valid Wikipedia fallback snapshot found.")
        ),
    )
    monkeypatch.setattr(
        "src.ftse350_history.OFFICIAL_CONSTITUENT_URLS",
        {
            "FTSE 100": ["https://example.com/ftse100"],
            "FTSE 250": ["https://example.com/ftse250"],
        },
    )

    with pytest.raises(SnapshotNotFoundError):
        build_history(
            years=[2025],
            out_dir=tmp_path,
            refresh_web=True,
            max_snapshot_gap_days=45,
            validate_deltas=False,
            write_raw=False,
            review_notices_json=None,
        )


def test_build_history_falls_back_to_wikipedia_on_underfilled_official_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyResponse:
        def __init__(self, *, json_payload=None, text="", status_code=200):
            self._json_payload = json_payload
            self.text = text
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._json_payload

    class DummySession:
        def get(self, url, params=None, timeout=None):
            if "wayback/available" in url:
                target = params["url"]
                segment = "ftse100" if "ftse100" in target else "ftse250"
                return DummyResponse(
                    json_payload={
                        "archived_snapshots": {
                            "closest": {
                                "available": True,
                                "timestamp": "20251231120000",
                                "url": f"https://web.archive.org/{segment}",
                            }
                        }
                    }
                )
            if "ftse100" in url:
                return DummyResponse(text=FTSE100_HTML.format(rows=make_rows("AAA", 20)))
            if "ftse250" in url:
                return DummyResponse(text=FTSE250_HTML.format(rows=make_rows("BBB", 20)))
            raise AssertionError(f"Unexpected URL {url}")

    def fake_resolve_wikipedia_fallback_snapshot(*args, **kwargs):
        segment = kwargs["segment"]
        row_count = 100 if segment == "FTSE 100" else 250
        prefix = "WAA" if segment == "FTSE 100" else "WBB"
        html = FTSE100_HTML.format(rows=make_rows(prefix, row_count))
        rows = parse_constituents_html(
            html,
            year=kwargs["year"],
            segment=segment,
            source_type="fallback_secondary",
            source_url=f"https://en.wikipedia.org/wiki/{segment.replace(' ', '_')}",
            archive_url=f"https://en.wikipedia.org/w/index.php?oldid={segment.replace(' ', '').lower()}",
            snapshot_date="2025-12-31",
            effective_date=kwargs["effective_date"],
            approximate_snapshot=False,
        )
        snapshot = {
            "year": kwargs["year"],
            "segment": segment,
            "source_url": f"https://en.wikipedia.org/wiki/{segment.replace(' ', '_')}",
            "archive_url": f"https://en.wikipedia.org/w/index.php?oldid={segment.replace(' ', '').lower()}",
            "snapshot_date": "2025-12-31",
            "approximate_snapshot": False,
            "source_type": "fallback_secondary",
            "html_path": None,
        }
        from src.ftse350_history import ArchivedSnapshot

        return ArchivedSnapshot(**snapshot), rows

    monkeypatch.setattr("src.ftse350_history.build_session", lambda: DummySession())
    monkeypatch.setattr(
        "src.ftse350_history.resolve_wikipedia_fallback_snapshot",
        fake_resolve_wikipedia_fallback_snapshot,
    )
    monkeypatch.setattr(
        "src.ftse350_history.OFFICIAL_CONSTITUENT_URLS",
        {
            "FTSE 100": ["https://example.com/ftse100"],
            "FTSE 250": ["https://example.com/ftse250"],
        },
    )

    result = build_history(
        years=[2025],
        out_dir=tmp_path,
        refresh_web=True,
        max_snapshot_gap_days=45,
        validate_deltas=False,
        write_raw=False,
        review_notices_json=None,
    )

    assert result["rows_written"] == 350
