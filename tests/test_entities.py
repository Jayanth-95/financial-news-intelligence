"""Regression tests for entity/ticker extraction (audit findings D2, D4, D5).

Covers: noisy company extraction, possessive normalization, false-positive
ticker acronyms, validated bare/parenthetical tickers, and TickerResolver's
conservative resolution order. Also checks percentage/currency/event
extraction is unchanged.

Run with: PYTHONPATH=. python3 tests/test_entities.py
"""
from __future__ import annotations

from app.services import TickerResolver
from utils.entities import is_valid_company_name, normalize_company_name
from utils.ner import extract_orgs
from utils.parse import (
    extract_companies,
    extract_currency_values,
    extract_events,
    extract_percentages,
    extract_tickers,
)


# --- 1. Noisy company extraction -----------------------------------------

def test_case_1_group_holdings_noise():
    text = "The Group announced changes. Also Apple Inc reported earnings. A separate Holdings update followed."
    companies = extract_companies(text)
    assert "Apple Inc" in companies, companies
    assert "The Group" not in companies, companies
    assert "Holdings" not in companies, companies
    assert "Also Apple Inc" not in companies, companies
    assert companies == ["Apple Inc"], companies


# --- 2 & 3. Possessive normalization --------------------------------------

def test_case_2_dell_possessive():
    assert extract_companies("Dell's earnings beat estimates.") == ["Dell"]
    assert extract_companies("Dell\u2019s earnings beat estimates.") == ["Dell"]


def test_case_3_hdfc_bank_possessive():
    assert extract_companies("HDFC Bank's revenue increased.") == ["HDFC Bank"]


def test_possessive_normalization_helper():
    assert normalize_company_name("Dell\u2019s") == "Dell"
    assert normalize_company_name("Lenovo's") == "Lenovo"
    assert normalize_company_name("HDFC Bank's") == "HDFC Bank"
    assert normalize_company_name("Apple Inc's") == "Apple Inc"
    # Internal apostrophes are not trailing possessives and must survive.
    assert normalize_company_name("O'Reilly") == "O'Reilly"


# --- 4. $AAPL cashtag ------------------------------------------------------

def test_case_4_cashtag():
    assert extract_tickers("$AAPL rose after earnings.") == ["AAPL"]


# --- 5-7. Financial acronym false positives -------------------------------

def test_case_5_eps_rejected():
    assert extract_tickers("EPS beat estimates.") == []


def test_case_6_ev_rejected():
    assert extract_tickers("EV adoption increased.") == []


def test_case_7_gdp_rejected():
    assert extract_tickers("GDP growth accelerated.") == []


def test_all_named_acronyms_rejected():
    for acronym in ("EPS", "IPO", "CEO", "CFO", "GDP", "YOY", "QOQ", "ETF", "SEC", "ESG", "EV", "FY", "AI"):
        text = f"{acronym} was mentioned in the report."
        assert extract_tickers(text) == [], f"{acronym} should not be extracted as a ticker"
        # Even parenthetically, and even if someone's mapping mistakenly
        # "validates" it -- the reject list takes precedence.
        paren_text = f"Something ({acronym}) happened."
        assert extract_tickers(paren_text, known_tickers={acronym}) == [], acronym


# --- 8. Validated parenthetical ticker ------------------------------------

def test_case_8_parenthetical_requires_validation():
    text = "Shares of Acme (ACME) rose."
    assert extract_tickers(text) == [], "ACME is not a known ticker without a mapping"
    assert extract_tickers(text, known_tickers={"ACME"}) == ["ACME"]


# --- 9. No arbitrary bare-uppercase tickers -------------------------------

def test_case_9_no_arbitrary_bare_tickers():
    tickers = extract_tickers("CS TECH Ai Q1-FY27")
    assert "CS" not in tickers, tickers
    assert "TECH" not in tickers, tickers
    assert "FY" not in tickers, tickers
    assert "FY27" not in tickers, tickers
    assert "Q1" not in tickers, tickers
    assert tickers == [], tickers


# --- 10. EV in a longer sentence -------------------------------------------

def test_case_10_ev_in_sentence():
    text = "Two Outperform-rated Chinese EV stocks climbed after the announcement."
    assert "EV" not in extract_tickers(text)


# --- Existing bare-ticker spec (was the broken test, D2) -------------------

def test_known_bare_ticker_still_recognized():
    # AAPL is a real, well-known ticker in the built-in KNOWN_TICKERS set,
    # so it's recognized in bare form -- this is the behavior the original
    # (broken) test_processing.py expected.
    assert "AAPL" in extract_tickers("AAPL gained 12.5% after earnings.")


# --- Generic-term rejection is structural, not a giant blacklist ----------

def test_generic_terms_rejected_directly():
    for term in ("Group", "Holdings", "Company", "Companies", "Market", "Markets",
                 "Investors", "Investor", "Industry", "Industries", "Inc", "Corp"):
        assert not is_valid_company_name(term), term


def test_leading_stopwords_rejected_as_standalone_candidates():
    for word in ("The", "Also", "A", "An", "This"):
        assert not is_valid_company_name(word), word


# --- spaCy path stays consistent with the regex path ----------------------

def test_ner_path_present_and_consistent_when_unavailable():
    # In this environment spaCy/en_core_web_sm may not be installed; extract_orgs
    # must degrade to an empty list rather than raising, and must never
    # bypass the shared validity/normalization rules when it does run.
    result = extract_orgs("The Group announced changes. Apple Inc reported earnings.")
    assert isinstance(result, list)
    assert "The Group" not in result
    assert "Group" not in result


# --- Percentages / currency / events unaffected ----------------------------

def test_percentage_currency_event_extraction_unchanged():
    text = "AAPL gained 12.5% after earnings. Revenue reached $2,500."
    assert extract_percentages(text) == ["12.5%"]
    assert extract_currency_values(text) == ["$2,500"]
    assert "earnings" in extract_events(text)


# --- TickerResolver: exact / alias / validated / bounded-fuzzy / unresolved

def test_resolver_exact_match():
    resolver = TickerResolver({"Acme Corp": "ACME"})
    assert resolver.resolve(["Acme Corp"]) == ["ACME"]


def test_resolver_alias_match_via_possessive_normalization():
    resolver = TickerResolver({"Dell": "DELL"})
    # "Dell's" normalizes to "Dell", matching the mapping via the alias path.
    assert resolver.resolve(["Dell's"]) == ["DELL"]


def test_resolver_accepts_candidate_that_is_already_a_valid_ticker():
    resolver = TickerResolver({})
    assert resolver.resolve(["IBM"]) == ["IBM"]  # in the built-in KNOWN_TICKERS set


def test_resolver_never_invents_a_ticker_for_unknown_short_or_generic_input():
    resolver = TickerResolver({"Acme Corp": "ACME"})
    assert resolver.resolve(["Holdings"]) == []
    assert resolver.resolve(["Xyz"]) == []  # short, unrelated, no fuzzy match should fire
    assert resolver.resolve([""]) == []


def test_resolver_fuzzy_match_requires_high_confidence():
    resolver = TickerResolver({"Reliance Industries": "RELIANCE"})
    # Near-exact typo should still resolve at a high cutoff.
    assert resolver.resolve(["Reliance Industriess"]) == ["RELIANCE"]
    # A loosely related but meaningfully different name should not.
    assert resolver.resolve(["Reliance Capital Ventures"]) == []


TESTS = [
    test_case_1_group_holdings_noise,
    test_case_2_dell_possessive,
    test_case_3_hdfc_bank_possessive,
    test_possessive_normalization_helper,
    test_case_4_cashtag,
    test_case_5_eps_rejected,
    test_case_6_ev_rejected,
    test_case_7_gdp_rejected,
    test_all_named_acronyms_rejected,
    test_case_8_parenthetical_requires_validation,
    test_case_9_no_arbitrary_bare_tickers,
    test_case_10_ev_in_sentence,
    test_known_bare_ticker_still_recognized,
    test_generic_terms_rejected_directly,
    test_leading_stopwords_rejected_as_standalone_candidates,
    test_ner_path_present_and_consistent_when_unavailable,
    test_percentage_currency_event_extraction_unchanged,
    test_resolver_exact_match,
    test_resolver_alias_match_via_possessive_normalization,
    test_resolver_accepts_candidate_that_is_already_a_valid_ticker,
    test_resolver_never_invents_a_ticker_for_unknown_short_or_generic_input,
    test_resolver_fuzzy_match_requires_high_confidence,
]


def run():
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    run()
