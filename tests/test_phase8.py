"""Phase 8 regression tests: recall failures (Bodycote, Shein, ServiceTitan,
Cytokinetics, Alkermes), person/pronoun false positives (Trump, Warsh,
Parton, She, "Following Warsh"), and publisher-vs-primary-subject ticker
weighting (The Block / Solana / Strive).

FinBERT is untouched in this phase -- see tests/test_sentiment.py.

Run with: PYTHONPATH=. python3 tests/test_phase8.py
"""
from __future__ import annotations

from unittest.mock import patch

import utils.ner as ner_mod
from tests._helpers import TICKER_MAPPING_PATH
from app.services import (
    ArticleAnalyzer,
    ArticleExtractor,
    AnalysisOptions,
    SentimentService,
    TickerResolver,
    load_ticker_mapping,
)
from utils.entities import is_valid_company_name
from utils.ner import merge_and_filter_companies
from utils.parse import extract_companies, extract_marked_tickers


class _FakeSpan:
    def __init__(self, text, label):
        self.text = text
        self.label_ = label


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class _FakeNLP:
    """Realistic-enough mock: only returns entities whose text actually
    occurs in the slice being parsed, so position-dependent behavior
    (primary-zone weighting) can be tested meaningfully.
    """

    def __init__(self, all_ents):
        self._all_ents = all_ents

    def __call__(self, text):
        return _FakeDoc([e for e in self._all_ents if e.text in text])


def _with_fake_spacy(ents):
    return patch.object(ner_mod, "_load_model", return_value=_FakeNLP(ents))


class _OfflineExtractor(ArticleExtractor):
    def __init__(self, text):
        self._text = text

    def extract(self, item):
        return self._text


class _FakeItem:
    def __init__(self, source="cnbc", title="", summary="", url="", published=""):
        self.source = source
        self.title = title
        self.summary = summary
        self.url = url
        self.published = published


def _resolver():
    mapping = load_ticker_mapping(TICKER_MAPPING_PATH)
    return TickerResolver(mapping)


def _analyzer(resolver, text, use_spacy=True):
    return ArticleAnalyzer(
        _OfflineExtractor(text), resolver, SentimentService(enabled=True, mock=True),
        AnalysisOptions(use_spacy=use_spacy),
    )


# =====================================================================
# False negatives -- legitimate companies must be recognized, no ticker
# required
# =====================================================================

def test_bodycote_and_veritas_recognized():
    text = "Bodycote agrees GBP 1.84bn takeover by US buyout firm Veritas, the companies said in a stock market filing."
    companies = extract_companies(text)
    assert "Bodycote" in companies
    assert "Veritas" in companies


def test_shein_recognized_without_requiring_a_ticker():
    text = "Shein shares slide 10% in Hong Kong trading debut on the stock market."
    companies = extract_companies(text)
    assert companies == ["Shein"]
    resolver = _resolver()
    assert resolver.resolve(["Shein"]) == []  # private -- no ticker, and none invented


def test_servicetitan_cytokinetics_alkermes_recognized():
    cases = {
        "ServiceTitan stock rating reiterated at Buy by TD Cowen analysts.": "ServiceTitan",
        "Cytokinetics shares rose after the company announced trial results.": "Cytokinetics",
        "Alkermes reported strong quarterly earnings and raised guidance.": "Alkermes",
    }
    for text, expected in cases.items():
        companies = extract_companies(text)
        assert expected in companies, (text, companies)


def test_full_pipeline_recall_headline_only_text():
    # "must identify ServiceTitan even if the body is only the short
    # headline/summary" -- use title+summary as the entire extracted text,
    # simulating a blocked full-article fetch.
    resolver = _resolver()
    analyzer = _analyzer(resolver, "ServiceTitan stock rating reiterated at Buy.", use_spacy=False)
    item = _FakeItem(title="ServiceTitan stock rating reiterated at Buy", summary="")
    result = analyzer.analyze(item)
    assert "ServiceTitan" in result.companies
    assert result.mapped_tickers == ["TTAN"]


# =====================================================================
# People and pronouns must never become companies
# =====================================================================

def test_trump_possessive_not_a_company_when_spacy_available():
    # The regex possessive pattern alone would produce "Trump" (from
    # "Trump's"); cross-checking against spaCy's PERSON tag for the same
    # text is what rejects it -- this is the core architectural fix.
    text = "The missing piece in Trump's oil and gas boom: jobs and market gains."
    regex_companies = extract_companies(text)
    assert "Trump" in regex_companies  # confirms the regex alone can't tell
    with _with_fake_spacy([_FakeSpan("Trump", "PERSON")]):
        merged = merge_and_filter_companies(text, regex_companies)
    assert "Trump" not in merged


def test_warsh_and_following_warsh_not_companies():
    text = "Following Warsh's remarks, CME Group markets reacted."
    ents = [_FakeSpan("Warsh", "PERSON"), _FakeSpan("CME Group", "ORG")]
    with _with_fake_spacy(ents):
        merged = merge_and_filter_companies(text, extract_companies(text))
    assert "Warsh" not in merged
    assert "Following Warsh" not in merged
    assert "CME Group" in merged


def test_parton_and_pronoun_she_not_companies():
    assert not is_valid_company_name("She")
    text = "Parton discussed her business ventures at CrowdStrike's conference."
    # Simulates spaCy inconsistently tagging "Parton" as both ORG and
    # PERSON in the same document (the real reported failure mode).
    ents = [_FakeSpan("Parton", "ORG"), _FakeSpan("Parton", "PERSON"), _FakeSpan("CrowdStrike", "ORG")]
    with _with_fake_spacy(ents):
        merged = merge_and_filter_companies(text, extract_companies(text))
    assert "Parton" not in merged
    assert "CrowdStrike" in merged


def test_first_index_investment_trust_wordplay_rejected():
    # "First Index Investment Trust" itself ends in "Trust" -- already
    # rejected by the existing institution-suffix rule.
    assert not is_valid_company_name("First Index Investment Trust")


# =====================================================================
# Publisher self-reference vs. actual financial subject (ticker weighting)
# =====================================================================

def test_block_publisher_disclaimer_does_not_create_sq():
    resolver = _resolver()
    title = "Solana price surges as trading market activity grows"
    summary = "Solana stock-like token trading activity increased sharply."
    body = (
        "Solana price surges as trading market activity grows. "
        + "Solana network trading activity increased sharply this week. " * 30
        + "This article was written by The Block. The Block requires readers "
        "to agree to its disclaimer. Block content is provided as-is."
    )
    analyzer = _analyzer(resolver, body)
    ents = [_FakeSpan("Solana", "ORG"), _FakeSpan("Block", "ORG"), _FakeSpan("The Block", "ORG")]
    with _with_fake_spacy(ents):
        result = analyzer.analyze(_FakeItem(source="the_block", title=title, summary=summary))
    assert "SQ" not in result.mapped_tickers, result.mapped_tickers


def test_strive_publisher_disclaimer_does_not_create_sq():
    resolver = _resolver()
    title = "Strive files for a new market fund amid trading interest"
    summary = "Strive fund activity increased this week in the stock market."
    body = (
        "Strive files for a new market fund amid trading interest. "
        + "Strive fund activity increased this week in the stock market. " * 30
        + "This article was written by The Block. The Block requires readers "
        "to agree to its disclaimer. Block content is provided as-is."
    )
    analyzer = _analyzer(resolver, body)
    ents = [_FakeSpan("Strive", "ORG"), _FakeSpan("Block", "ORG"), _FakeSpan("The Block", "ORG")]
    with _with_fake_spacy(ents):
        result = analyzer.analyze(_FakeItem(source="the_block", title=title, summary=summary))
    assert "SQ" not in result.mapped_tickers, result.mapped_tickers


def test_block_inc_explicitly_identified_as_subject_resolves_to_sq():
    resolver = _resolver()
    title = "Block Inc. shares rise after earnings beat"
    summary = "Block Inc. (SQ) posted strong quarterly results in the stock market."
    body = "Block Inc. (SQ) shares rise after the company posted a strong earnings beat this quarter."
    analyzer = _analyzer(resolver, body)
    with _with_fake_spacy([_FakeSpan("Block Inc", "ORG")]):
        result = analyzer.analyze(_FakeItem(title=title, summary=summary))
    assert "SQ" in result.mapped_tickers


def test_strive_explicit_ticker_annotation_resolves():
    text = "Strive (ASST) shares climbed in stock market trading today."
    assert extract_marked_tickers(text, known_tickers={"ASST"}) == ["ASST"]
    resolver = _resolver()
    analyzer = _analyzer(resolver, text)
    with _with_fake_spacy([_FakeSpan("Strive", "ORG")]):
        result = analyzer.analyze(_FakeItem(title="Strive (ASST) shares climb", summary=text))
    assert "ASST" in result.mapped_tickers


# =====================================================================
# Regression: core companies from earlier phases still resolve
# =====================================================================

def test_core_companies_still_resolve():
    resolver = _resolver()
    assert resolver.resolve(["Berkshire Hathaway Inc"]) == ["BRK.B"]
    assert resolver.resolve(["Nvidia"]) == ["NVDA"]
    assert resolver.resolve(["Salesforce"]) == ["CRM"]


TESTS = [
    test_bodycote_and_veritas_recognized,
    test_shein_recognized_without_requiring_a_ticker,
    test_servicetitan_cytokinetics_alkermes_recognized,
    test_full_pipeline_recall_headline_only_text,
    test_trump_possessive_not_a_company_when_spacy_available,
    test_warsh_and_following_warsh_not_companies,
    test_parton_and_pronoun_she_not_companies,
    test_first_index_investment_trust_wordplay_rejected,
    test_block_publisher_disclaimer_does_not_create_sq,
    test_strive_publisher_disclaimer_does_not_create_sq,
    test_block_inc_explicitly_identified_as_subject_resolves_to_sq,
    test_strive_explicit_ticker_annotation_resolves,
    test_core_companies_still_resolve,
]


def run():
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    run()
