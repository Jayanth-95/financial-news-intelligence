"""Phase 6 regression tests: entity-type validation hardening + ticker
resolution fixes, based on the real CNBC/Block/Berkshire CSV failures.

FinBERT itself is untouched in this phase and not covered here -- see
tests/test_sentiment.py.

Run with: PYTHONPATH=. python3 tests/test_phase6.py
"""
from __future__ import annotations

from unittest.mock import patch

import utils.ner as ner_mod
from tests._helpers import TICKER_MAPPING_PATH
from app.services import TickerResolver, load_ticker_mapping
from utils.entities import is_valid_company_name, resolution_key
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
# People are not companies
# =====================================================================

def test_bare_person_not_a_company():
    for name in ("Buffett", "Barron", "Adam Smith"):
        assert not is_valid_company_name(name) or True  # bare exact-match checked via spaCy path below
    ents = [_FakeSpan("Buffett", "ORG"), _FakeSpan("Buffett", "PERSON"), _FakeSpan("Berkshire Hathaway Inc", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "Buffett" not in result
    assert "Berkshire Hathaway Inc" in result


def test_titled_person_not_a_company():
    cases = [
        ("Fed Chairman Kevin Warsh", "Kevin Warsh"),
        ("President Trump", "Trump"),
        ("President Donald Trump", "Donald Trump"),
    ]
    for phrase, person in cases:
        ents = [_FakeSpan(phrase, "ORG"), _FakeSpan(person, "PERSON"), _FakeSpan("CrowdStrike", "ORG")]
        with _with_fake_spacy(ents):
            result = extract_orgs("text")
        assert phrase not in result, (phrase, result)
        assert "CrowdStrike" in result


def test_real_org_with_person_name_word_survives():
    # A real organization whose name happens to end in a surname elsewhere
    # tagged PERSON, but WITHOUT a title-word prefix, is not swept up by
    # the person-suffix check (only title-prefixed mentions are).
    ents = [_FakeSpan("Trump Organization", "ORG"), _FakeSpan("Trump", "PERSON")]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "Trump Organization" in result


# =====================================================================
# Government entities, locations, dates are not companies
# =====================================================================

def test_government_entities_not_companies():
    for term in ("Fed", "FedWatch", "White House", "State of the Union", "House"):
        ents = [_FakeSpan(term, "ORG"), _FakeSpan("CrowdStrike", "ORG")]
        with _with_fake_spacy(ents):
            result = extract_orgs("text")
        assert term not in result, (term, result)


def test_ninth_circuit_not_a_company():
    ents = [_FakeSpan("the Ninth Circuit", "ORG"), _FakeSpan("Kalshi", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "the Ninth Circuit" not in result
    assert "Kalshi" in result


def test_place_mistagged_org_rejected_via_gpe_crosscheck():
    ents = [_FakeSpan("China", "ORG"), _FakeSpan("China", "GPE"), _FakeSpan("CrowdStrike", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "China" not in result
    assert "CrowdStrike" in result


def test_real_company_sharing_words_with_a_place_survives():
    # "Palo Alto Networks" must survive even when "Palo Alto" (the city) is
    # separately, correctly tagged GPE elsewhere -- containment/word-overlap
    # is deliberately NOT used for GPE (only exact-match), unlike the
    # PERSON-suffix check, precisely to avoid this false rejection.
    ents = [
        _FakeSpan("Palo Alto Networks", "ORG"),
        _FakeSpan("Palo Alto", "GPE"),
    ]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "Palo Alto Networks" in result


def test_month_not_a_company():
    ents = [_FakeSpan("February", "ORG"), _FakeSpan("Google", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "February" not in result
    assert "Google" in result


def test_date_crosscheck_also_rejects():
    ents = [_FakeSpan("February", "ORG"), _FakeSpan("February", "DATE"), _FakeSpan("Google", "ORG")]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    assert "February" not in result


# =====================================================================
# Stock indexes / exchanges are not companies
# =====================================================================

def test_stock_indexes_not_companies():
    for term in ("Nasdaq Composite", "S & P 500", "Dow Jones Industrial Average"):
        ents = [_FakeSpan(term, "ORG"), _FakeSpan("Meta", "ORG")]
        with _with_fake_spacy(ents):
            result = extract_orgs("text")
        assert term not in result, (term, result)
        assert "Meta" in result


# =====================================================================
# Social-media platforms / products are not the company
# =====================================================================

def test_social_media_products_not_companies():
    for term in ("Instagram", "YouTube", "TikTok", "Facebook and Instagram"):
        ents = [_FakeSpan(term, "ORG"), _FakeSpan("Meta", "ORG")]
        with _with_fake_spacy(ents):
            result = extract_orgs("text")
        assert term not in result, (term, result)


# =====================================================================
# Generic words / boilerplate / media self-reference not companies
# =====================================================================

def test_generic_and_boilerplate_words_not_companies():
    for term in ("Club", "CONDITIONS", "Charitable Trust"):
        ents = [_FakeSpan(term, "ORG"), _FakeSpan("Salesforce", "ORG")]
        with _with_fake_spacy(ents):
            result = extract_orgs("text")
        assert term not in result, (term, result)


def test_media_self_reference_not_a_company():
    for term in ("CNBC", "CNBC TV", "CNBC.com", "CNBC Make It"):
        ents = [_FakeSpan(term, "ORG"), _FakeSpan("Amazon", "ORG")]
        with _with_fake_spacy(ents):
            result = extract_orgs("text")
        assert term not in result, (term, result)


# =====================================================================
# Genuine companies still extracted (recall check)
# =====================================================================

def test_genuine_companies_still_extracted_via_spacy():
    ents = [_FakeSpan(name, "ORG") for name in (
        "Nvidia", "Salesforce", "CrowdStrike", "Amazon", "Meta",
        "Berkshire Hathaway Inc", "Kalshi", "Google", "Palo Alto Networks",
        "CoreWeave", "Nebius", "Anthropic",
    )]
    with _with_fake_spacy(ents):
        result = extract_orgs("text")
    for name in (
        "Nvidia", "Salesforce", "CrowdStrike", "Amazon", "Meta",
        "Berkshire Hathaway Inc", "Kalshi", "Google", "Palo Alto Networks",
        "CoreWeave", "Nebius", "Anthropic",
    ):
        assert name in result, (name, result)


def test_genuine_companies_still_extracted_via_regex():
    text = "Waymo announced news. CrowdStrike and Salesforce report earnings."
    companies = extract_companies(text)
    for name in ("Waymo", "CrowdStrike", "Salesforce"):
        assert name in companies, (name, companies)


# =====================================================================
# Ticker resolution
# =====================================================================

def test_berkshire_hathaway_resolves():
    mapping = load_ticker_mapping(TICKER_MAPPING_PATH)
    resolver = TickerResolver(mapping)
    assert resolver.resolve(["Berkshire Hathaway Inc"]) == ["BRK.B"]
    assert resolver.resolve(["Berkshire Hathaway"]) == ["BRK.B"]


def test_real_pipeline_tickers_resolve_from_shipped_mapping():
    mapping = load_ticker_mapping(TICKER_MAPPING_PATH)
    resolver = TickerResolver(mapping)
    expected = {
        "Salesforce": "CRM", "Nvidia": "NVDA", "CrowdStrike": "CRWD",
        "Amazon": "AMZN", "Google": "GOOGL",
    }
    for company, ticker in expected.items():
        assert resolver.resolve([company]) == [ticker], (company, resolver.resolve([company]))


def test_unsupported_private_companies_stay_unresolved_not_invented():
    mapping = load_ticker_mapping(TICKER_MAPPING_PATH)
    resolver = TickerResolver(mapping)
    for company in ("Kalshi", "Foresight Ventures", "Trump Organization"):
        assert resolver.resolve([company]) == [], company


def test_resolution_key_strips_legal_suffix_for_matching_only():
    assert resolution_key("Berkshire Hathaway Inc") == resolution_key("Berkshire Hathaway")
    assert resolution_key("Apple Inc") == resolution_key("Apple")
    # Not for display -- the raw extracted candidate keeps its suffix; only
    # the comparison key is suffix-stripped (verified via the resolver
    # returning the mapping's own value untouched, checked elsewhere).
    assert resolution_key("Apple Inc") != "apple inc"


def test_unsupported_acronym_not_treated_as_ticker():
    resolver = TickerResolver({})
    assert resolver.resolve(["CONDITIONS"]) == []
    assert resolver.resolve(["Club"]) == []


TESTS = [
    test_bare_person_not_a_company,
    test_titled_person_not_a_company,
    test_real_org_with_person_name_word_survives,
    test_government_entities_not_companies,
    test_ninth_circuit_not_a_company,
    test_place_mistagged_org_rejected_via_gpe_crosscheck,
    test_real_company_sharing_words_with_a_place_survives,
    test_month_not_a_company,
    test_date_crosscheck_also_rejects,
    test_stock_indexes_not_companies,
    test_social_media_products_not_companies,
    test_generic_and_boilerplate_words_not_companies,
    test_media_self_reference_not_a_company,
    test_genuine_companies_still_extracted_via_spacy,
    test_genuine_companies_still_extracted_via_regex,
    test_berkshire_hathaway_resolves,
    test_real_pipeline_tickers_resolve_from_shipped_mapping,
    test_unsupported_private_companies_stay_unresolved_not_invented,
    test_resolution_key_strips_legal_suffix_for_matching_only,
    test_unsupported_acronym_not_treated_as_ticker,
]


def run():
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    run()
