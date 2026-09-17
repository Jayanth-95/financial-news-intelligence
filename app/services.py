"""Application services for article extraction, analysis, and enrichment."""
from __future__ import annotations

import csv
import difflib
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from utils.entities import KNOWN_TICKERS, is_valid_company_name, normalize_company_name, resolution_key
from utils.extract import clean_text, extract_main_text
from utils.filtering import is_stock_market_related
from utils.ner import extract_orgs, merge_and_filter_companies
from utils.network import fetch_with_timeout
from utils.parse import extract_financial_entities, extract_marked_tickers
from utils.sentiment import finbert_infer
from .models import ArticleAnalysis, NewsItem, Sentiment

LOG = logging.getLogger(__name__)

_BOILERPLATE_MARKERS = (
    "register now", "read this article for free", "read free articles",
    "explore more offers", "for individuals", "follow topics", "personalised alerts",
    "access alphaville", "our popular markets and finance blog",
)


def article_key(item: NewsItem) -> str:
    raw = f"{item.source}::{item.title}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ArticleExtractor:
    def extract(self, item: NewsItem) -> str:
        fallback = clean_text(f"{item.title}. {item.summary}".strip())
        if not item.url:
            return fallback
        html = fetch_with_timeout(item.url)
        if not html:
            return fallback
        extracted = clean_text(extract_main_text(html) or "")
        if not extracted or self._looks_like_boilerplate(extracted, item):
            return fallback
        return extracted

    @staticmethod
    def _looks_like_boilerplate(text: str, item: NewsItem) -> bool:
        lower = text.casefold()
        marker_hits = sum(1 for marker in _BOILERPLATE_MARKERS if marker in lower)
        title_words = [w for w in re.findall(r"[a-z]{4,}", item.title.casefold())]
        title_overlap = sum(1 for word in title_words if word in lower)
        # A short extraction dominated by subscription/navigation language is not
        # useful for NER or sentiment, so fall back to the source metadata.
        if marker_hits >= 2 and len(text) < 2500:
            return True
        if marker_hits >= 3 and title_overlap < max(2, len(title_words) // 3):
            return True
        return False


class TickerResolver:
    """Resolves extracted company-name candidates to ticker symbols.

    Resolution is deliberately conservative -- an unresolved candidate is
    dropped, never guessed. Priority order:
      1. Exact match against the canonical company -> ticker mapping.
      2. Alias match: the mapping key after the same normalization
         (possessive/whitespace stripping) applied to the candidate, so
         "Dell's" resolves the same way "Dell" does.
      3. Suffix-normalized alias match: both sides with a trailing
         legal-entity suffix stripped (see utils.entities.resolution_key),
         so "Berkshire Hathaway Inc" (as extracted, suffix included)
         matches a mapping keyed just "Berkshire Hathaway", and vice versa,
         without needing every suffix variant listed in the mapping file.
      4. The candidate is itself already a known ticker symbol (from the
         loaded mapping's own tickers, or the small built-in KNOWN_TICKERS
         set), e.g. an extractor that surfaced "IBM" directly.
      5. A tightly-bounded fuzzy match against the mapping -- only for
         longer, already-validated company-name candidates, and only at a
         high similarity threshold. Short or generic candidates never reach
         this step, since noisy input to difflib produces noisy tickers.
    Anything not resolved by one of these is left out, not guessed at.
    """

    def __init__(self, mapping: dict[str, str], cutoff: float = 0.92):
        self.mapping = {key.casefold(): value.strip() for key, value in mapping.items() if key and value}
        self.alias_mapping = {
            normalize_company_name(key).casefold(): value for key, value in self.mapping.items()
        }
        self.resolution_index = {
            resolution_key(key): value for key, value in mapping.items() if key and value
        }
        self.known_tickers = {value.upper() for value in self.mapping.values()} | KNOWN_TICKERS
        # Original-cased company names from the mapping, so extract_companies()
        # can recognize a bare mention of a mapped company (e.g. "Waymo") by
        # exact-case whole-word matching against the article text -- casefold
        # keys in self.mapping lose the casing needed for that.
        self.display_names = {key.strip() for key in mapping if key and key.strip()}
        self.cutoff = cutoff

    def resolve(
        self,
        companies: Sequence[str],
        *,
        primary_companies: Sequence[str] | None = None,
        explicit_tickers: Sequence[str] | None = None,
    ) -> list[str]:
        """Resolve company candidates to tickers.

        `primary_companies`, if given, restricts full name-based resolution
        (tiers 1-5 above) to that subset -- e.g. companies found in the
        article's title/summary/lede, rather than anywhere in the full
        text. This is what stops a publisher self-reference or disclaimer
        mention (e.g. "The Block" appearing only in footer text, alongside
        a bare "Block" mention elsewhere in boilerplate) from justifying a
        ticker association just because the company name occurs somewhere
        in the scraped text. If omitted, behavior is unchanged: every
        candidate in `companies` is resolved (backward compatible).

        `explicit_tickers` are tickers backed by a strong, position-
        independent textual marker (a cashtag, a parenthetical annotation,
        an "NYSE: X" / "trades under X" phrase -- see
        utils.parse.extract_marked_tickers). These are included regardless
        of position, since someone explicitly writing a ticker as a ticker
        is strong evidence on its own -- but only if the symbol is one this
        resolver actually recognizes (known mapping ticker or built-in
        KNOWN_TICKERS), never invented.
        """
        keys = list(self.mapping)
        out = set()
        for company in (primary_companies if primary_companies is not None else companies):
            ticker = self._resolve_one(company, keys)
            if ticker:
                out.add(ticker)
        if explicit_tickers:
            out |= {t.strip().upper() for t in explicit_tickers if t.strip().upper() in self.known_tickers}
        return sorted(out)

    def _resolve_one(self, company: str, keys: list[str]):
        candidate = normalize_company_name(company)
        if not is_valid_company_name(candidate):
            return None
        key = candidate.casefold()

        ticker = self.mapping.get(key) or self.alias_mapping.get(key)
        if ticker:
            return ticker.strip()

        ticker = self.resolution_index.get(resolution_key(candidate))
        if ticker:
            return ticker.strip()

        upper = candidate.upper()
        if upper in self.known_tickers:
            return upper

        if len(candidate) >= 4 and keys:
            match = difflib.get_close_matches(key, keys, n=1, cutoff=self.cutoff)
            if match:
                return self.mapping[match[0]].strip()

        return None


class SentimentService:
    """The pipeline's sentiment entry point. Delegates all real inference to
    utils.sentiment.finbert_infer -- the single authoritative, local FinBERT
    implementation. There is no fallback to a different model: a load or
    inference failure is reported as a controlled "error" sentiment state,
    never silently answered by a different model's output.
    """

    def __init__(self, enabled: bool = True, mock: bool = False):
        self.enabled = enabled
        self.mock = mock

    def analyze(self, text: str) -> Sentiment:
        if not self.enabled:
            return Sentiment("skipped", 0.0)
        if self.mock:
            return Sentiment("neutral", 0.0)
        try:
            result = finbert_infer(text)
        except Exception as exc:
            LOG.warning("FinBERT sentiment inference failed: %s", exc)
            return Sentiment("error", 0.0)
        LOG.debug("Sentiment scored by %s: %s (%.4f)", result.model, result.label, result.score)
        return Sentiment(result.label, result.score)


@dataclass(frozen=True)
class AnalysisOptions:
    use_spacy: bool = True


class ArticleAnalyzer:
    # How much of the extracted article body counts as the "primary zone"
    # (along with the title and summary, always included) for ticker
    # resolution -- see analyze(). Matches the same "the article states its
    # actual subject near the top" assumption already used for event
    # extraction's lede window.
    _PRIMARY_ZONE_CHARS = 1000

    def __init__(self, extractor, resolver, sentiment, options):
        self.extractor = extractor
        self.resolver = resolver
        self.sentiment = sentiment
        self.options = options

    def analyze(self, item: NewsItem):
        title = clean_text(item.title)
        summary = clean_text(item.summary)
        if not title or not is_stock_market_related(f"{title} {summary}"):
            return None

        text = self.extractor.extract(item)
        if not is_stock_market_related(text):
            return None

        entities = extract_financial_entities(
            text,
            known_tickers=self.resolver.known_tickers,
            known_company_names=self.resolver.display_names,
        )
        companies = self._merge_spacy(text, list(entities.get("companies", [])))

        if item.source:
            source_name = clean_text(item.source).casefold()
            companies = [
                company
                for company in companies
                if company.casefold() != source_name
            ]

        primary_text = f"{title} {summary} {text[:self._PRIMARY_ZONE_CHARS]}".strip()
        if primary_text == text:
            primary_companies = companies
        else:
            primary_entities = extract_financial_entities(
                primary_text,
                known_tickers=self.resolver.known_tickers,
                known_company_names=self.resolver.display_names,
            )
            primary_companies = self._merge_spacy(primary_text, list(primary_entities.get("companies", [])))
            companies = list(dict.fromkeys(
    primary_companies + companies
))
            

        explicit_tickers = extract_marked_tickers(text, known_tickers=self.resolver.known_tickers)

        normalized = NewsItem(item.source, title, summary, item.url, item.published)
        return ArticleAnalysis(
            normalized,
            text,
            companies,
            entities.get("tickers", []),
            self.resolver.resolve(companies, primary_companies=primary_companies, explicit_tickers=explicit_tickers),
            entities.get("percentages", []),
            entities.get("currency_values", []),
            entities.get("events", []),
            self.sentiment.analyze(text),
        )

    def _merge_spacy(self, text: str, companies: list[str]) -> list[str]:
        """Merge in (and cross-check, when spaCy is enabled) spaCy's own ORG
        output for the same text -- see utils.ner.merge_and_filter_companies
        for why the cross-check applies to the regex candidates too, not
        just spaCy's.
        """
        if not self.options.use_spacy:
            return companies
        return merge_and_filter_companies(text, companies, known_tickers=self.resolver.known_tickers)


def load_ticker_mapping(path):
    if not path:
        LOG.warning(
            "No ticker map path provided; ticker resolution will fall back to the "
            "built-in known-ticker set only (most companies will be unmapped)."
        )
        return {}
    if not Path(path).exists():
        LOG.warning(
            "Ticker map not found at %s (cwd=%s); ticker resolution will fall back "
            "to the built-in known-ticker set only (most companies will be unmapped). "
            "If running pipeline.py from a directory other than the project root, "
            "pass --ticker-map with an explicit path.",
            path, Path.cwd(),
        )
        return {}
    result = {}
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for index, row in enumerate(csv.reader(handle)):
                if len(row) < 2:
                    continue
                if index == 0 and row[0].strip().casefold() == "company":
                    continue
                company, ticker = row[0].strip(), row[1].strip()
                if company and ticker:
                    result[company] = ticker
    except (OSError, csv.Error) as exc:
        LOG.warning("Ticker map unavailable: %s", exc)
    if not result:
        LOG.warning("Ticker map at %s loaded but contained no usable rows.", path)
    else:
        LOG.info("Loaded %d ticker mapping(s) from %s", len(result), path)
    return result
