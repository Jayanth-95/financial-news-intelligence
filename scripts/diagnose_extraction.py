"""Diagnose the real ArticleAnalyzer -> extractor -> TickerResolver call
chain, stage by stage, for a set of articles. Uses the REAL production
classes (ArticleAnalyzer, TickerResolver, load_ticker_mapping,
merge_and_filter_companies) -- nothing reimplemented -- so what this prints
is exactly what the real pipeline does, not an approximation of it.

Run with:
    python -m scripts.diagnose_extraction
    python scripts/diagnose_extraction.py

For each article, prints:
    A) title / summary / text actually received
    B) regex-derived company candidates (utils.parse.extract_companies)
    C) spaCy's own ORG candidates for the same text (raw, pre-filter)
    D) final filtered/merged companies (utils.ner.merge_and_filter_companies)
    E) what's handed to TickerResolver.resolve() (companies, primary_companies,
       explicit_tickers)
    F) TickerResolver's raw per-candidate resolution attempts
    G) final mapped_tickers

Also reports, up front, the two confirmed silent-failure points this
script exists to catch:
    - Whether data/ticker_mapping.csv actually loaded (and from where)
    - Whether spaCy's real en_core_web_sm model loaded, or silently fell
      back to a no-op blank pipeline, or isn't installed at all

The article bodies below are RECONSTRUCTED approximations of the headlines
reported as failing, built to be structurally representative (same
company names, same kind of surrounding text) -- not verbatim scraped
text, since that wasn't available to this script. Swap in real scraped
text (paste it into ARTICLES below) for a byte-exact reproduction of a
specific failure.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# See scripts/finbert_smoke_test.py for why this is needed for direct
# (`python scripts/diagnose_extraction.py`) invocation to work without an
# installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

import pipeline
from app.services import (
    AnalysisOptions,
    ArticleAnalyzer,
    ArticleExtractor,
    SentimentService,
    TickerResolver,
    load_ticker_mapping,
)
from app.models import NewsItem
from utils.ner import _load_model, merge_and_filter_companies
from utils.parse import extract_companies, extract_financial_entities, extract_marked_tickers


class _FixedTextExtractor(ArticleExtractor):
    """Stands in for the real network-backed extractor: returns exactly
    the text given, so this script tests entity/ticker logic in isolation
    from network/scraping behavior.
    """

    def __init__(self, text: str):
        self._text = text

    def extract(self, item):
        return self._text


ARTICLES = {
    "FT Trump article": {
        "title": "The missing piece in Trump's oil and gas boom: jobs",
        "summary": "Analysts say Trump's energy policy has boosted output but not employment in the sector.",
        "body": (
            "The missing piece in Trump's oil and gas boom: jobs. Analysts say Trump's "
            "energy policy has boosted output but not employment in the sector, according "
            "to a new market report on drilling activity and stock performance."
        ),
    },
    "The Block Solana/publisher article": {
        "title": "Solana price surges as trading market activity grows",
        "summary": "Solana network trading activity increased sharply in the stock and crypto market this week.",
        "body": (
            "Solana price surges as trading market activity grows. "
            + "Solana network trading activity increased sharply in the stock and crypto market this week. " * 25
            + "This article was written by The Block. The Block requires readers to agree to "
            "its disclaimer. Block content is provided as-is. Foresight Ventures and Multicoin "
            "Capital were cited as sources."
        ),
    },
    "Anthropic IPO article": {
        "title": "Anthropic weighs IPO as Morgan Stanley and Goldman advise on market listing",
        "summary": "Morgan Stanley and Goldman Sachs are reportedly advising Anthropic on a stock market listing.",
        "body": (
            "Anthropic weighs IPO as Morgan Stanley and Goldman advise on market listing. "
            "Morgan Stanley and Goldman Sachs are reportedly advising Anthropic on a stock "
            "market listing that could value the AI company highly."
        ),
    },
    "Berkshire Hathaway article": {
        "title": "Berkshire Hathaway reports record quarterly profit in stock market filing",
        "summary": "Berkshire Hathaway posted record profit in its latest earnings and market filing.",
        "body": (
            "Berkshire Hathaway reports record quarterly profit in stock market filing. "
            "Berkshire Hathaway posted record profit in its latest earnings and market filing."
        ),
    },
    "Lululemon article": {
        "title": "Lululemon shares fall after weak stock market guidance",
        "summary": "Lululemon shares fell in trading after the company issued weak guidance to the market.",
        "body": (
            "Lululemon shares fall after weak stock market guidance. Lululemon shares fell "
            "in trading after the company issued weak guidance to the market."
        ),
    },
    "Tesla Robotaxi article": {
        "title": "Tesla Robotaxi expansion challenges Alphabet in the market",
        "summary": "Tesla and Alphabet's Waymo are competing for market share in autonomous ride-hailing stock coverage.",
        "body": (
            "Tesla Robotaxi expansion challenges Alphabet in the market. RBC Capital Markets "
            "and Wells Fargo weighed in on Tesla's stock as it competes with Alphabet's Waymo "
            "for autonomous ride-hailing market share."
        ),
    },
    "Investing.com Fastly article": {
        "title": "Fastly stock rating reiterated as FSLY shares trade in the market",
        "summary": "FSLY shares moved in stock market trading after an analyst rating update.",
        "body": "Fastly stock rating reiterated as FSLY shares trade in the market.",
    },
    "Investing.com PubMatic/Gevo insider-trading article": {
        "title": "PubMatic insider trading disclosed in stock market filing",
        "summary": "A PubMatic executive disclosed an insider stock market trade; Gevo also filed a related market report.",
        "body": "PubMatic insider trading disclosed in stock market filing. Gevo also filed a related market report.",
    },
}


def _print_header(label: str) -> None:
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)


def main() -> None:
    _print_header("0) Environment diagnosis (the two confirmed silent-failure points)")

    print(f"pipeline.PROJECT_ROOT      = {pipeline.PROJECT_ROOT}")
    print(f"pipeline.DEFAULT_TICKER_MAP = {pipeline.DEFAULT_TICKER_MAP}")
    print(f"Path exists?                = {Path(pipeline.DEFAULT_TICKER_MAP).exists()}")
    mapping = load_ticker_mapping(pipeline.DEFAULT_TICKER_MAP)
    print(f"Ticker mapping rows loaded  = {len(mapping)}")
    print(f"'Berkshire Hathaway' in map = {'Berkshire Hathaway' in mapping}")

    model = _load_model()
    if model is None:
        print("spaCy model status          = NOT LOADED (spaCy unavailable) -- regex-only extraction")
    elif getattr(model, "pipe_names", None):
        print(f"spaCy model status          = REAL model loaded, pipes={model.pipe_names}")
    else:
        print("spaCy model status          = BLANK fallback (en_core_web_sm missing) -- "
              "NO named-entity recognition is actually running")

    resolver = TickerResolver(mapping)
    analyzer = ArticleAnalyzer(
        _FixedTextExtractor(""), resolver, SentimentService(enabled=False), AnalysisOptions(use_spacy=True)
    )

    for name, article in ARTICLES.items():
        _print_header(name)
        title, summary, body = article["title"], article["summary"], article["body"]

        print("A) title  :", title)
        print("   summary:", summary)
        print("   text[:200]:", body[:200])

        regex_companies = extract_companies(
            body, known_company_names=resolver.display_names, known_tickers=resolver.known_tickers
        )
        print("B) regex candidates       :", regex_companies)

        if model is not None:
            doc = model(body)
            spacy_orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        else:
            spacy_orgs = ["<spaCy unavailable>"]
        print("C) spaCy raw ORG spans    :", spacy_orgs)

        filtered = merge_and_filter_companies(body, regex_companies, known_tickers=resolver.known_tickers)
        print("D) filtered/merged companies:", filtered)

        primary_text = f"{title} {summary} {body[:analyzer._PRIMARY_ZONE_CHARS]}".strip()
        primary_companies = merge_and_filter_companies(
            primary_text,
            extract_companies(primary_text, known_company_names=resolver.display_names, known_tickers=resolver.known_tickers),
            known_tickers=resolver.known_tickers,
        )
        explicit_tickers = extract_marked_tickers(body, known_tickers=resolver.known_tickers)
        print("E) resolver input:")
        print("     companies         =", filtered)
        print("     primary_companies =", primary_companies)
        print("     explicit_tickers  =", explicit_tickers)

        for company in filtered:
            print(f"F)   resolve[{company!r}] ->", resolver.resolve([company]))

        mapped = resolver.resolve(filtered, primary_companies=primary_companies, explicit_tickers=explicit_tickers)
        print("G) mapped_tickers (final) :", mapped)


if __name__ == "__main__":
    main()
