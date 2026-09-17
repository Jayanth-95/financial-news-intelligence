from __future__ import annotations
import pytest
@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path
"""Tests for PipelineRunner's output-file handling (app/runner.py).

Covers the data-loss fix: existing CSV rows must survive subsequent runs,
new rows must be appended and deduplicated, and a missing/empty/malformed
existing file must never crash the run or wipe a good dataset.

Uses a stub analyzer/source so no network, spaCy, or HF access is required.
Run with: PYTHONPATH=. python3 tests/test_runner.py
"""

import csv
import shutil
import tempfile
from pathlib import Path

from app.models import ArticleAnalysis, Sentiment
from app.runner import PipelineRunner


class StubAnalyzer:
    """Deterministic analyzer: returns one row per item, no extraction/sentiment work."""

    def analyze(self, item):
        return ArticleAnalysis(item=item, text=item.title, sentiment=Sentiment("neutral", 0.0))


def make_source(items, name="stub_source"):
    def _source(days=1):
        yield from items
    _source.__name__ = name
    return _source


def _item(title, source="stub"):
    return {"source": source, "title": title, "summary": "", "url": "", "published": ""}


def _read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _titles(path: Path):
    return sorted(r["title"] for r in _read_rows(path))


def test_first_run_creates_rows(tmp_dir: Path):
    out = tmp_dir / "run1.csv"
    a, b = _item("Article A"), _item("Article B")
    count = PipelineRunner([make_source([a, b])], StubAnalyzer(), out, days=1).execute()
    assert count == 2, f"expected 2 rows written, got {count}"
    assert _titles(out) == ["Article A", "Article B"]


def test_second_run_same_articles_no_duplicates(tmp_dir: Path):
    out = tmp_dir / "run2.csv"
    a, b = _item("Article A"), _item("Article B")
    PipelineRunner([make_source([a, b])], StubAnalyzer(), out, days=1).execute()
    count = PipelineRunner([make_source([a, b])], StubAnalyzer(), out, days=1).execute()
    assert count == 2, f"expected no duplicates, got {count} total rows"
    assert _titles(out) == ["Article A", "Article B"]


def test_second_run_new_articles_preserves_old_and_new(tmp_dir: Path):
    out = tmp_dir / "run3.csv"
    a, b, c = _item("Article A"), _item("Article B"), _item("Article C")
    PipelineRunner([make_source([a, b])], StubAnalyzer(), out, days=1).execute()
    count = PipelineRunner([make_source([c])], StubAnalyzer(), out, days=1).execute()
    assert count == 3, f"expected old(2)+new(1)=3 rows, got {count}"
    assert _titles(out) == ["Article A", "Article B", "Article C"]


def test_zero_new_article_run_does_not_erase_dataset(tmp_dir: Path):
    out = tmp_dir / "run4.csv"
    a, b, c = _item("Article A"), _item("Article B"), _item("Article C")
    PipelineRunner([make_source([a, b, c])], StubAnalyzer(), out, days=1).execute()
    # A run that yields nothing at all (e.g. every source failed/timed out).
    count = PipelineRunner([make_source([])], StubAnalyzer(), out, days=1).execute()
    assert count == 3, f"dataset must survive a zero-item run, got {count} rows"
    assert _titles(out) == ["Article A", "Article B", "Article C"]
    # A run where every item is a duplicate of what's already stored.
    count = PipelineRunner([make_source([a, b, c])], StubAnalyzer(), out, days=1).execute()
    assert count == 3, f"dataset must survive an all-duplicates run, got {count} rows"


def test_missing_output_file_is_safe(tmp_dir: Path):
    out = tmp_dir / "does_not_exist" / "out.csv"
    count = PipelineRunner([make_source([_item("Article A")])], StubAnalyzer(), out, days=1).execute()
    assert count == 1
    assert out.exists()


def test_empty_output_file_is_safe(tmp_dir: Path):
    out = tmp_dir / "empty.csv"
    out.write_text("", encoding="utf-8")
    count = PipelineRunner([make_source([_item("Article A")])], StubAnalyzer(), out, days=1).execute()
    assert count == 1
    assert _titles(out) == ["Article A"]


def test_malformed_schema_file_is_safe(tmp_dir: Path):
    out = tmp_dir / "malformed.csv"
    # Valid CSV, but not our schema at all (no source/title columns).
    out.write_text("foo,bar\n1,2\n", encoding="utf-8")
    count = PipelineRunner([make_source([_item("Article A")])], StubAnalyzer(), out, days=1).execute()
    assert count == 1, f"unrelated existing rows should be dropped, not crash; got {count}"
    assert _titles(out) == ["Article A"]


def test_garbage_binary_file_is_safe(tmp_dir: Path):
    out = tmp_dir / "garbage.csv"
    out.write_bytes(b"\xff\xfe\x00not-valid-utf8-or-csv\x00\xff")
    # Must not raise; must degrade to "start fresh" rather than crash the run.
    count = PipelineRunner([make_source([_item("Article A")])], StubAnalyzer(), out, days=1).execute()
    assert count == 1
    assert _titles(out) == ["Article A"]


def test_no_leftover_temp_files(tmp_dir: Path):
    out = tmp_dir / "atomic.csv"
    PipelineRunner([make_source([_item("Article A")])], StubAnalyzer(), out, days=1).execute()
    leftovers = list(tmp_dir.glob(".tmp-*"))
    assert not leftovers, f"atomic write left temp files behind: {leftovers}"


TESTS = [
    test_first_run_creates_rows,
    test_second_run_same_articles_no_duplicates,
    test_second_run_new_articles_preserves_old_and_new,
    test_zero_new_article_run_does_not_erase_dataset,
    test_missing_output_file_is_safe,
    test_empty_output_file_is_safe,
    test_malformed_schema_file_is_safe,
    test_garbage_binary_file_is_safe,
    test_no_leftover_temp_files,
]


def run():
    base = Path(tempfile.mkdtemp(prefix="runner_test_"))
    try:
        for test in TESTS:
            case_dir = base / test.__name__
            case_dir.mkdir()
            test(case_dir)
            print(f"PASS: {test.__name__}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    run()
