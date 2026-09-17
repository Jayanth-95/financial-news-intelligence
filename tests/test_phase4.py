"""Phase 4 regression tests: real-data quality / extraction hardening.

Covers the exact failures observed in the real 41-row production run:
company extraction (Dick's Sporting Goods, CrowdStrike/Salesforce, Waymo,
AMD/Intel/Nvidia, junk rejection), punctuation preservation, ticker mapping,
the --days publication-window filter (the 2024 Moneycontrol bug), Phase 1
persistence non-regression, and event-extraction quality.

None of this requires spaCy, torch, transformers, or network access --
the spaCy path is tested against a fake NLP pipeline that simulates
realistic ORG/PERSON tagging.

Run with: PYTHONPATH=. python3 tests/test_phase4.py
"""
from __future__ import annotations

import csv
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import utils.ner as ner_mod
from app.models import ArticleAnalysis, NewsItem, Sentiment
from app.runner import PipelineRunner
from app.services import TickerResolver
from utils.dates import is_within_window, parse_published
from utils.entities import is_valid_company_name, normalize_company_name
from utils.ner import extract_orgs
from utils.parse import extract_companies, extract_events


# =====================================================================
# 1. Company extraction -- the real observed failures (regex path)
# =====================================================================

def test_dicks_sporting_goods_preserved_whole():
    text = "Dick's Sporting Goods reported weaker-than-expected earnings amid a slowdown in discretionary spending."
    assert extract_companies(text) == ["Dick's Sporting Goods"]
    assert "Dick" not in extract_companies(text)


def test_crowdstrike_and_salesforce():
    text = "CrowdStrike and Salesforce report earnings Wednesday, says Jim Cramer on CNBC. AI remains a hot topic."
    companies = extract_companies(text)
    assert "CrowdStrike" in companies
    assert "Salesforce" in companies
    for junk in ("AI", "Wednesday", "Jim Cramer", "CNBC", "Jim"):
        assert junk not in companies, (junk, companies)


def test_waymo_bare_mention():
    text = "Waymo to launch driverless rides in Germany in 2027, expanding beyond the US market."
    assert extract_companies(text) == ["Waymo"]


def test_amd_intel_nvidia():
    text = "AMD can beat rivals Intel and Nvidia in the next generation of AI chips, analysts say."
    companies = extract_companies(text)
    for name in ("AMD", "Intel", "Nvidia"):
        assert name in companies, (name, companies)
    assert "AI" not in companies


def test_junk_never_becomes_a_company():
    cases = {
        "Tuesday market update.": "Tuesday",
        "Here is the report.": "Here",
        "There was no change.": "There",
        "CNBC reported today.": "CNBC",
        "Jim Cramer weighed in on the stock.": "Jim Cramer",
    }
    for text, junk in cases.items():
        companies = extract_companies(text)
        assert junk not in companies, (text, companies)


def test_temporal_phrase_and_conjunction_leading_junk_rejected():
    # Regression: raw NER-style spans like "This Thursday"/"Last Thursday"
    # were not caught by the old whole-string-only temporal check, and
    # "While High-Flyer" was not caught since "While" wasn't a leading
    # stopword and the check never looked at just the first word.
    for junk in ("This Thursday", "Last Thursday", "While High-Flyer"):
        assert not is_valid_company_name(junk), junk


def test_non_company_institution_suffix_rejected():
    assert not is_valid_company_name("Brookings Institution")
    assert not is_valid_company_name("Federal Reserve")
    # A genuine company name is unaffected.
    assert is_valid_company_name("CrowdStrike")


# =====================================================================
# 2. Punctuation / possessive preservation
# =====================================================================

def test_punctuated_names_survive_normalization():
    assert extract_companies("McDonald's reported strong same-store sales.") == ["McDonald's"]
    assert extract_companies("O'Reilly Automotive beat estimates.") == ["O'Reilly Automotive"]
    assert extract_companies("AT&T raised its dividend.") == ["AT&T"]
    assert normalize_company_name("McDonald's") == "McDonald's"
    assert normalize_company_name("Dick's Sporting Goods") == "Dick's Sporting Goods"
    assert normalize_company_name("O'Reilly") == "O'Reilly"
    assert normalize_company_name("AT&T") == "AT&T"


def test_phase2_possessive_regression_unaffected():
    # Generic (non-whitelisted) possessive companies still get their
    # possessive correctly stripped -- only the curated brand-name
    # whitelist is exempted.
    assert extract_companies("Dell's earnings beat estimates.") == ["Dell"]
    assert extract_companies("HDFC Bank's revenue increased.") == ["HDFC Bank"]


def test_phase2_noise_regression_unaffected():
    text = "The Group announced changes. Also Apple Inc reported earnings. A separate Holdings update followed."
    assert extract_companies(text) == ["Apple Inc"]


# =====================================================================
# 3. Ticker mapping
# =====================================================================

def test_resolver_maps_real_companies_to_real_tickers():
    mapping = {"CrowdStrike": "CRWD", "Salesforce": "CRM", "Intel": "INTC", "Nvidia": "NVDA"}
    resolver = TickerResolver(mapping)
    assert resolver.resolve(["CrowdStrike"]) == ["CRWD"]
    assert resolver.resolve(["Salesforce"]) == ["CRM"]
    assert resolver.resolve(["Intel"]) == ["INTC"]
    assert resolver.resolve(["Nvidia"]) == ["NVDA"]


def test_resolver_recognizes_bare_known_ticker_without_mapping():
    resolver = TickerResolver({})
    assert resolver.resolve(["AMD"]) == ["AMD"]


def test_resolver_leaves_unmapped_company_empty_never_invents():
    resolver = TickerResolver({"CrowdStrike": "CRWD"})
    assert resolver.resolve(["Waymo"]) == []


def test_resolver_display_names_preserve_original_casing_for_bare_matching():
    resolver = TickerResolver({"Waymo": "GOOGL_SUBSIDIARY_PLACEHOLDER"})
    assert "Waymo" in resolver.display_names


def test_ticker_mapping_file_exists_and_resolves_reported_companies():
    """Regression for the real root cause of 'ticker extraction essentially
    empty': data/ticker_mapping.csv never existed, so TickerResolver always
    ran with an empty mapping -- Nvidia/Salesforce/CrowdStrike/Apple etc.
    have no algorithmic way to become NVDA/CRM/CRWD/AAPL without a lookup.
    """
    from app.services import load_ticker_mapping
    path = Path(__file__).resolve().parents[1] / "data" / "ticker_mapping.csv"
    assert path.exists(), "data/ticker_mapping.csv must exist"
    mapping = load_ticker_mapping(str(path))
    assert mapping, "ticker mapping must not be empty"
    resolver = TickerResolver(mapping)
    for company, ticker in {
        "Nvidia": "NVDA", "Salesforce": "CRM", "CrowdStrike": "CRWD",
        "Apple": "AAPL", "Intel": "INTC", "AMD": "AMD",
    }.items():
        assert resolver.resolve([company]) == [ticker], (company, resolver.resolve([company]))


# =====================================================================
# 4. Contextual entity validation -- spaCy path (mocked NLP pipeline)
# =====================================================================

class _FakeSpan:
    def __init__(self, text, label):
        self.text = text
        self.label_ = label


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class _FakeNLP:
    def __init__(self, ents):
        self._ents = ents

    def __call__(self, text):
        return _FakeDoc(self._ents)


def _with_fake_spacy(ents):
    return patch.object(ner_mod, "_load_model", return_value=_FakeNLP(ents))


def test_spacy_path_accepts_genuine_companies():
    ents = [_FakeSpan("CrowdStrike", "ORG"), _FakeSpan("Salesforce", "ORG"), _FakeSpan("Waymo", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("CrowdStrike and Salesforce report earnings; Waymo also announced news.")
    assert set(result) == {"CrowdStrike", "Salesforce", "Waymo"}


def test_spacy_path_rejects_weekday_and_deictic_org_tags():
    ents = [_FakeSpan("Wednesday", "ORG"), _FakeSpan("Here", "ORG"), _FakeSpan("CrowdStrike", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("some text")
    assert "Wednesday" not in result
    assert "Here" not in result
    assert "CrowdStrike" in result


def test_spacy_path_rejects_bare_acronym_shaped_org_not_a_known_ticker():
    ents = [_FakeSpan("AI", "ORG"), _FakeSpan("CNBC", "ORG"), _FakeSpan("IBM", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("some text")
    assert "AI" not in result
    assert "CNBC" not in result
    assert "IBM" in result  # a genuine known ticker/company, correctly kept


def test_spacy_path_rejects_org_that_is_also_tagged_person():
    # Simulates spaCy inconsistently tagging "Jim Cramer" as both ORG and
    # PERSON in the same document -- the PERSON tag should win.
    ents = [
        _FakeSpan("Jim Cramer", "ORG"),
        _FakeSpan("Jim Cramer", "PERSON"),
        _FakeSpan("CrowdStrike", "ORG"),
    ]
    with _with_fake_spacy(ents):
        result = extract_orgs("Jim Cramer discussed CrowdStrike on air.")
    assert "Jim Cramer" not in result
    assert "CrowdStrike" in result


def test_spacy_path_rejects_org_that_is_also_tagged_place():
    # "China" mistagged ORG in one place but correctly tagged GPE
    # elsewhere in the same document -- GPE should win.
    ents = [
        _FakeSpan("China", "ORG"),
        _FakeSpan("China", "GPE"),
        _FakeSpan("CrowdStrike", "ORG"),
    ]
    with _with_fake_spacy(ents):
        result = extract_orgs("China's market reacted; CrowdStrike also reported.")
    assert "China" not in result
    assert "CrowdStrike" in result


def test_spacy_path_degrades_gracefully_when_model_unavailable():
    with patch.object(ner_mod, "_load_model", return_value=None):
        assert extract_orgs("CrowdStrike reported earnings.") == []


# =====================================================================
# 5. --days publication-window filter
# =====================================================================

_NOW = datetime(2026, 8, 28, tzinfo=timezone.utc)


def test_2024_article_rejected_by_days_1_in_2026():
    # The exact reported production bug.
    assert is_within_window("Tue, 23 Apr 2024 10:15:00 +0530", 1, now=_NOW) is False


def test_article_inside_window_accepted():
    assert is_within_window("Fri, 28 Aug 2026 09:00:00 +0000", 1, now=_NOW) is True


def test_article_outside_window_rejected():
    assert is_within_window("Tue, 25 Aug 2026 09:00:00 +0000", 1, now=_NOW) is False


def test_timezone_conversion_is_correct():
    # 2026-08-28T04:00:00+05:30 == 2026-08-27T22:30:00 UTC -- within 1 day of _NOW.
    assert is_within_window("2026-08-28T04:00:00+05:30", 1, now=_NOW) is True
    # A naive timestamp is assumed UTC, not silently guessed as local time.
    assert is_within_window("2026-08-27T23:00:00", 1, now=_NOW) is True
    assert parse_published("2026-08-27T23:00:00").tzinfo is not None


def test_iso_z_format_reddit_style():
    assert is_within_window("2026-08-27T20:00:00Z", 1, now=_NOW) is True
    assert is_within_window("2024-04-23T10:15:00Z", 1, now=_NOW) is False


def test_malformed_date_fails_open():
    assert is_within_window("not a date at all", 1, now=_NOW) is True
    assert parse_published("not a date at all") is None


def test_missing_date_fails_open():
    assert is_within_window("", 1, now=_NOW) is True
    assert is_within_window(None, 1, now=_NOW) is True


# =====================================================================
# 6. Persistence non-regression (Phase 1) + duplicate prevention, now
#    combined with the date filter actually being active
# =====================================================================

class _StubAnalyzer:
    def analyze(self, item):
        return ArticleAnalysis(item=item, text=item.title, sentiment=Sentiment("neutral", 0.0))


def _make_source(items):
    def _source(days=1):
        yield from items
    _source.__name__ = "stub_source"
    return _source


def _item(title, published, source="moneycontrol"):
    return {"source": source, "title": title, "summary": "", "url": "", "published": published}


def _read_titles(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row["title"] for row in csv.DictReader(f)]


def _recent(hours=1):
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%a, %d %b %Y %H:%M:%S +0000")


def test_historical_rows_preserved_and_stale_new_articles_excluded():
    out = Path(tempfile.mkdtemp()) / "out.csv"
    fresh = _item("Fresh Article", _recent(1))
    stale = _item("Patel Engineering", "Tue, 23 Apr 2024 10:15:00 +0530")

    first = PipelineRunner([_make_source([fresh, stale])], _StubAnalyzer(), out, days=1).execute()
    assert first == 1  # only the fresh article was accepted
    titles = _read_titles(out)
    assert "Patel Engineering" not in titles
    assert "Fresh Article" in titles

    # A second run with a different fresh article: old row preserved, new
    # row added, still no 2024 leakage.
    another_fresh = _item("Another Fresh Article", _recent(2))
    second = PipelineRunner([_make_source([another_fresh, stale])], _StubAnalyzer(), out, days=1).execute()
    assert second == 2
    titles = _read_titles(out)
    assert set(titles) == {"Fresh Article", "Another Fresh Article"}


def test_running_same_articles_twice_does_not_duplicate():
    out = Path(tempfile.mkdtemp()) / "out.csv"
    fresh = _item("Repeatable Article", _recent(1))
    PipelineRunner([_make_source([fresh])], _StubAnalyzer(), out, days=1).execute()
    count = PipelineRunner([_make_source([fresh])], _StubAnalyzer(), out, days=1).execute()
    assert count == 1


# =====================================================================
# 7. Event extraction quality
# =====================================================================

def test_event_in_lede_is_kept():
    assert "earnings" in extract_events("Apple reported strong earnings this quarter.")


def test_event_repeated_is_kept():
    text = "A merger was announced today. Investors reacted to the merger news quickly."
    assert "merger" in extract_events(text)


def test_event_mentioned_once_deep_in_article_is_dropped():
    padding = "The market moved on broad economic news today. " * 20  # >600 chars
    text = padding + "Somewhere deep in this article, a merger was mentioned exactly once in passing."
    assert "merger" not in extract_events(text)


TESTS = [
    test_dicks_sporting_goods_preserved_whole,
    test_crowdstrike_and_salesforce,
    test_waymo_bare_mention,
    test_amd_intel_nvidia,
    test_junk_never_becomes_a_company,
    test_temporal_phrase_and_conjunction_leading_junk_rejected,
    test_non_company_institution_suffix_rejected,
    test_punctuated_names_survive_normalization,
    test_phase2_possessive_regression_unaffected,
    test_phase2_noise_regression_unaffected,
    test_resolver_maps_real_companies_to_real_tickers,
    test_resolver_recognizes_bare_known_ticker_without_mapping,
    test_resolver_leaves_unmapped_company_empty_never_invents,
    test_resolver_display_names_preserve_original_casing_for_bare_matching,
    test_ticker_mapping_file_exists_and_resolves_reported_companies,
    test_spacy_path_accepts_genuine_companies,
    test_spacy_path_rejects_weekday_and_deictic_org_tags,
    test_spacy_path_rejects_bare_acronym_shaped_org_not_a_known_ticker,
    test_spacy_path_rejects_org_that_is_also_tagged_person,
    test_spacy_path_rejects_org_that_is_also_tagged_place,
    test_spacy_path_degrades_gracefully_when_model_unavailable,
    test_2024_article_rejected_by_days_1_in_2026,
    test_article_inside_window_accepted,
    test_article_outside_window_rejected,
    test_timezone_conversion_is_correct,
    test_iso_z_format_reddit_style,
    test_malformed_date_fails_open,
    test_missing_date_fails_open,
    test_historical_rows_preserved_and_stale_new_articles_excluded,
    test_running_same_articles_twice_does_not_duplicate,
    test_event_in_lede_is_kept,
    test_event_repeated_is_kept,
    test_event_mentioned_once_deep_in_article_is_dropped,
]


def run():
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    run()
