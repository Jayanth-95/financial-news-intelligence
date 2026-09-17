"""Phase 7 regression tests: round-2 real-CSV extraction failures (Exxon,
Broadcom, Berkshire Hathaway comma-suffix resolution, malformed
parenthetical ticker spans).

FinBERT is untouched in this phase -- see tests/test_sentiment.py.

Run with: PYTHONPATH=. python3 tests/test_phase7.py
"""
from __future__ import annotations

from unittest.mock import patch

import utils.ner as ner_mod
from tests._helpers import TICKER_MAPPING_PATH
from app.services import TickerResolver, load_ticker_mapping
from utils.entities import normalize_company_name, resolution_key
from utils.ner import extract_orgs
from utils.parse import extract_companies


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


# =====================================================================
# Berkshire Hathaway ticker resolution -- the real reported regression
# =====================================================================

def test_berkshire_hathaway_resolves_with_comma_suffix_convention():
    # "Berkshire Hathaway, Inc." (comma before the suffix) is the common US
    # corporate-naming convention and is what broke resolution previously.
    mapping = load_ticker_mapping(TICKER_MAPPING_PATH)
    resolver = TickerResolver(mapping)
    for variant in (
        "Berkshire Hathaway Inc",
        "Berkshire Hathaway, Inc.",
        "Berkshire Hathaway Inc.",
        "Berkshire Hathaway",
    ):
        assert resolver.resolve([variant]) == ["BRK.B"], variant


def test_resolution_key_handles_comma_before_suffix():
    assert resolution_key("Berkshire Hathaway, Inc.") == resolution_key("Berkshire Hathaway")
    assert resolution_key("Exxon Mobil, Corp.") == resolution_key("Exxon Mobil")


def test_exxon_mobil_resolves_both_spelling_variants():
    mapping = load_ticker_mapping(TICKER_MAPPING_PATH)
    resolver = TickerResolver(mapping)
    assert resolver.resolve(["Exxon Mobil"]) == ["XOM"]
    assert resolver.resolve(["ExxonMobil"]) == ["XOM"]


# =====================================================================
# XOM must not appear in `companies` -- a ticker symbol is not a name
# =====================================================================

def test_bare_ticker_not_treated_as_company_name():
    text = "Exxon Mobil reported strong earnings. Shares of XOM rose 3%."
    companies = extract_companies(text, known_company_names={"Exxon Mobil"}, known_tickers={"XOM"})
    assert "XOM" not in companies
    assert "Exxon Mobil" in companies


def test_amd_still_recognized_as_company_and_ticker_doubling_as_name():
    # AMD is the deliberate exception: it IS commonly spoken as the
    # company's own name, unlike XOM/NVDA/CRWD/etc.
    assert extract_companies("AMD reported record revenue this quarter.") == ["AMD"]


# =====================================================================
# Exxon article: malformed spans rejected, legitimate companies kept
# =====================================================================

def test_exxon_article_malformed_spans_rejected():
    ents = [
        _FakeSpan("Exxon Mobil", "ORG"),
        _FakeSpan("Diamondback Energy", "ORG"),
        _FakeSpan("Diamondback Energy Mehta", "ORG"),
        _FakeSpan("Mehta", "PERSON"),
        _FakeSpan("Exxon Mobil Integrated", "ORG"),
        _FakeSpan("Exxon Mobil Stock Buybacks on TipRanks", "ORG"),
        _FakeSpan("See Expand Energy Insider Trading on", "ORG"),
        _FakeSpan("Expand Energy", "ORG"),
        _FakeSpan("Diamondback Energy Financials on TipRanks", "ORG"),
        _FakeSpan("Morgan Stanley", "ORG"),
        _FakeSpan("Goldman Sachs", "ORG"),
        _FakeSpan("TipRanks", "ORG"),
    ]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    for junk in (
        "Exxon Mobil Integrated", "Exxon Mobil Stock Buybacks on TipRanks",
        "See Expand Energy Insider Trading on", "Diamondback Energy Financials on TipRanks",
        "Diamondback Energy Mehta",
    ):
        assert junk not in result, (junk, result)
    for good in ("Exxon Mobil", "Diamondback Energy", "Expand Energy", "Morgan Stanley", "Goldman Sachs", "TipRanks"):
        assert good in result, (good, result)


# =====================================================================
# Broadcom article: title+name (self-contained), timezone glue,
# business-suffix trailing-person, boilerplate/institution junk
# =====================================================================

def test_title_then_name_rejected_without_needing_separate_person_tag():
    # Self-contained: no separate "George Kurtz" PERSON entity needed --
    # the title word "CEO" immediately followed by two capitalized words
    # is itself sufficient signal.
    ents = [_FakeSpan("CrowdStrike CEO George Kurtz", "ORG"), _FakeSpan("CrowdStrike", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "CrowdStrike CEO George Kurtz" not in result
    assert "CrowdStrike" in result


def test_timezone_abbreviation_glued_to_company_rejected():
    ents = [_FakeSpan("ET CrowdStrike", "ORG"), _FakeSpan("CrowdStrike", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "ET CrowdStrike" not in result
    assert "CrowdStrike" in result


def test_business_suffix_trailing_person_rejected_without_title_word():
    # No title word present ("Diamondback Energy Mehta") -- the signal here
    # is that "Diamondback Energy" (everything before the trailing PERSON
    # match) already ends in a business-type word ("Energy").
    ents = [
        _FakeSpan("Diamondback Energy Mehta", "ORG"),
        _FakeSpan("Mehta", "PERSON"),
        _FakeSpan("Diamondback Energy", "ORG"),
    ]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "Diamondback Energy Mehta" not in result
    assert "Diamondback Energy" in result


def test_bare_surname_contained_in_longer_person_span_rejected():
    ents = [_FakeSpan("Warsh", "ORG"), _FakeSpan("Kevin Warsh", "PERSON"), _FakeSpan("Broadcom", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "Warsh" not in result
    assert "Broadcom" in result


def test_supply_management_institute_with_internal_for_rejected():
    ents = [
        _FakeSpan("Institute for Supply Management's Manufacturing PMI", "ORG"),
        _FakeSpan("Broadcom", "ORG"),
    ]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "Institute for Supply Management's Manufacturing PMI" not in result
    assert "Broadcom" in result


def test_charitable_trust_and_boilerplate_rejected():
    ents = [
        _FakeSpan("Charitable Trust", "ORG"),
        _FakeSpan("CNBC TV", "ORG"),
        _FakeSpan("CONDITIONS", "ORG"),
        _FakeSpan("Broadcom", "ORG"),
    ]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    for junk in ("Charitable Trust", "CNBC TV", "CONDITIONS"):
        assert junk not in result, (junk, result)
    assert "Broadcom" in result


def test_legitimate_broadcom_companies_all_survive():
    ents = [_FakeSpan(name, "ORG") for name in (
        "Broadcom", "Palo Alto Networks", "Nvidia", "CrowdStrike", "Google",
        "Meta Platforms", "OpenAI", "Marvell Technology", "RBC Capital Markets",
        "UBS", "JPMorgan", "Intel", "Nio", "Snowflake", "Okta", "SentinelOne",
        "CyberArk", "Dell Technologies",
    )]
    # In the real pipeline, ArticleAnalyzer passes resolver.known_tickers
    # (built-in KNOWN_TICKERS plus every ticker in the loaded mapping file)
    # through to extract_orgs -- mirror that here rather than testing
    # extract_orgs in isolation from the ticker context it actually runs with.
    mapping = load_ticker_mapping(TICKER_MAPPING_PATH)
    resolver = TickerResolver(mapping)
    with _with_fake_spacy(ents):
        result = extract_orgs("text", known_tickers=resolver.known_tickers)
    for name in (
        "Broadcom", "Palo Alto Networks", "Nvidia", "CrowdStrike", "Google",
        "Meta Platforms", "OpenAI", "Marvell Technology", "RBC Capital Markets",
        "UBS", "JPMorgan", "Intel", "Nio", "Snowflake", "Okta", "SentinelOne",
        "CyberArk", "Dell Technologies",
    ):
        assert name in result, (name, result)


# =====================================================================
# Malformed parenthetical ticker spans
# =====================================================================

def test_well_formed_ticker_annotation_cleaned_not_rejected():
    assert normalize_company_name("Nio (NIO)") == "Nio"


def test_unbalanced_parenthesis_spans_rejected_outright():
    ents = [
        _FakeSpan("Nio (NIO", "ORG"),
        _FakeSpan("NIO (NIO, Medtronic", "ORG"),
        _FakeSpan("Nio (NIO)", "ORG"),
        _FakeSpan("Medtronic", "ORG"),
    ]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "Nio (NIO" not in result
    assert "NIO (NIO, Medtronic" not in result
    assert "Nio" in result
    assert "Medtronic" in result


TESTS = [
    test_berkshire_hathaway_resolves_with_comma_suffix_convention,
    test_resolution_key_handles_comma_before_suffix,
    test_exxon_mobil_resolves_both_spelling_variants,
    test_bare_ticker_not_treated_as_company_name,
    test_amd_still_recognized_as_company_and_ticker_doubling_as_name,
    test_exxon_article_malformed_spans_rejected,
    test_title_then_name_rejected_without_needing_separate_person_tag,
    test_timezone_abbreviation_glued_to_company_rejected,
    test_business_suffix_trailing_person_rejected_without_title_word,
    test_bare_surname_contained_in_longer_person_span_rejected,
    test_supply_management_institute_with_internal_for_rejected,
    test_charitable_trust_and_boilerplate_rejected,
    test_legitimate_broadcom_companies_all_survive,
    test_well_formed_ticker_annotation_cleaned_not_rejected,
    test_unbalanced_parenthesis_spans_rejected_outright,
]


def run():
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    run()
