"""Deterministic extraction of financial entities from text."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from utils.entities import (
    KNOWN_BARE_COMPANY_NAMES,
    KNOWN_TICKERS,
    LEADING_STOPWORDS,
    REJECTED_TICKER_ACRONYMS,
    TICKER_DOUBLES_AS_COMPANY_NAME,
    is_valid_company_name,
    normalize_company_name,
)

_PERCENTAGE_RE = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_CURRENCY_RE = re.compile(
    r"(?<!\w)(?:(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?|"
    r"(?:\$|USD)\s*[\d,]+(?:\.\d+)?|"
    r"(?:€|EUR)\s*[\d,]+(?:\.\d+)?|"
    r"(?:£|GBP)\s*[\d,]+(?:\.\d+)?)(?!\w)",
    re.IGNORECASE,
)

# --- Tickers -----------------------------------------------------------
#
# A cashtag ($AAPL) is an explicit, unambiguous marker -- the "$" itself is
# the signal of intent, so it's accepted without further validation (beyond
# the acronym backstop below).
#
# Parenthetical ("Acme (ACME)") and bare ("AAPL rose") mentions carry no
# such marker, so they are only accepted when the candidate is a *known*
# ticker symbol. This is a default-deny design: an unrecognized all-caps
# token is reported as "not a ticker" rather than guessed at. Unknown
# tickers are correctly treated as unresolved, not invented.
_EXPLICIT_TICKER_RE = re.compile(r"(?<![A-Za-z0-9])\$([A-Z]{1,6})(?![A-Za-z0-9])")
_PAREN_TICKER_RE = re.compile(r"\(([A-Z]{1,6})\)")
# Bare (unmarked) tickers require at least 2 letters: a single stray capital
# letter is far too ambiguous in ordinary prose (initials, roman numerals,
# sentence artifacts) to treat as a cashtag-free ticker mention. Marked
# forms ($X, (X)) keep full 1-6 length support since the marker itself
# disambiguates intent.
_BARE_TICKER_RE = re.compile(r"(?<![A-Za-z0-9$])[A-Z]{2,6}(?![A-Za-z0-9])")

# Explicit phrase patterns that name a ticker directly, regardless of where
# in the article they occur -- "NYSE: SQ", "trades under the ticker ASST",
# "ticker symbol NVDA". These are used as strong, position-independent
# evidence for ticker association (see extract_marked_tickers), distinct
# from a bare mention of the same letters anywhere in running text.
_EXCHANGE_TICKER_RE = re.compile(r"\b(?:NYSE|Nasdaq|NASDAQ)\s*:\s*([A-Z]{1,6})\b")
_TRADES_UNDER_RE = re.compile(r"(?i:trades? under(?: the ticker)? )([A-Z]{1,6})\b")
_TICKER_SYMBOL_RE = re.compile(r"(?i:ticker(?: symbol)?\s*(?:is|was|:|=)?\s+)([A-Z]{1,6})\b")

# --- Companies -----------------------------------------------------------
#
# A "name token" is a capitalized word that is not a closed-class function
# word (see LEADING_STOPWORDS). Excluding those words from the token pattern
# itself -- rather than trimming them off after the fact -- is what stops a
# leading "The"/"Also" from ever entering a match: the regex engine simply
# can't start a name run there, so it advances to the real name. Internal
# apostrophes/ampersands are allowed in the token itself (not just as a
# trailing marker) so names like "O'Reilly", "Dick's", and "AT&T" are single
# tokens, not broken at the punctuation.
_STOPWORD_ALTERNATION = "|".join(re.escape(w) for w in sorted(LEADING_STOPWORDS, key=len, reverse=True))
_NAME_TOKEN = rf"(?!(?:{_STOPWORD_ALTERNATION})\b)[A-Z][A-Za-z&.\-\u2019']{{1,30}}"

_LEGAL_SUFFIXES = (
    "Inc\\.?", "Corp\\.?", "Corporation", "Ltd\\.?", "Limited", "PLC",
    "Holdings", "Group", "Technologies", "Technology",
)

# Legal-entity-suffix companies: "Apple Inc", "Reliance Industries Group".
# Requires at least one real name token before the suffix -- a bare
# "Holdings" or "Group" with nothing in front of it can never match, which
# is what keeps standalone generic suffix words out of the results.
_COMPANY_SUFFIX_RE = re.compile(
    rf"\b(?:{_NAME_TOKEN}\s+){{1,4}}(?:{'|'.join(_LEGAL_SUFFIXES)})\b"
)

# Possessive-anchored companies: "Dell's earnings", "HDFC Bank's revenue".
# Financial writing very commonly names a company as the possessive subject
# of a sentence even without a legal suffix; the possessive marker is a
# genuine structural signal that the preceding proper-noun phrase is a named
# entity, not an arbitrary keyword match.
_COMPANY_POSSESSIVE_RE = re.compile(
    rf"\b{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}}[\u2019']s\b"
)

# Bare (unmarked) proper-noun phrases -- candidate generator only. A match
# here is never accepted on pattern alone; it's only kept if it (or a
# leading sub-phrase of it) is in the known bare-company whitelist -- see
# extract_companies(). This is what lets "Waymo announced..." or
# "CrowdStrike and Salesforce report..." be recognized without a legal
# suffix or possessive marker, while still being default-deny (an unknown
# capitalized phrase is never accepted just because it matched this).
_BARE_PHRASE_RE = re.compile(rf"\b{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}}\b")

_EVENT_TERMS = (
    "earnings", "guidance", "ipo", "dividend", "buyback", "upgrade",
    "downgrade", "merger", "acquisition", "layoff", "profit", "loss", "forecast",
)
# A term mentioned once, deep in a long article, is a weaker signal that the
# article is actually *about* that event than one appearing in the lede
# (where news articles conventionally state what happened) or one repeated
# more than once. This keeps a single passing mention from being reported
# as an "event" without needing per-article scoring/ML.
_EVENT_LEDE_WINDOW = 600


@dataclass(frozen=True)
class FinancialEntities:
    companies: list[str]
    tickers: list[str]
    percentages: list[str]
    currency_values: list[str]
    events: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        value = value.strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _drop_shorter_overlaps(values: list[str]) -> list[str]:
    """Drop a candidate that is wholly contained in a longer candidate.

    Two of our own patterns can both fire on the same mention at different
    granularities -- e.g. the possessive pattern matching just "Dick's"
    inside "Dick's Sporting Goods" while the bare-phrase whitelist path
    matches the full name. Keep the longer, more complete one.
    """
    result = []
    for value in values:
        folded = value.casefold()
        if any(folded != other.casefold() and folded in other.casefold() for other in values):
            continue
        result.append(value)
    return result


def extract_percentages(text: str | None) -> list[str]:
    return _unique(_PERCENTAGE_RE.findall(text or ""))


def extract_currency_values(text: str | None) -> list[str]:
    return _unique(_CURRENCY_RE.findall(text or ""))


def extract_tickers(text: str | None, known_tickers: Iterable[str] | None = None) -> list[str]:
    """Extract ticker symbols, never inventing one.

    `known_tickers` lets a caller (e.g. ArticleAnalyzer, wired to a real
    ticker/company mapping) extend recognition of bare/parenthetical
    mentions beyond the built-in KNOWN_TICKERS seed list. A parenthetical
    or bare candidate is only accepted if it's a member of that combined
    set; an explicit cashtag ($AAPL) is accepted on its own signal. Common
    financial acronyms are rejected either way.
    """
    source = text or ""
    valid = KNOWN_TICKERS | {t.strip().upper() for t in (known_tickers or ()) if t}
    valid -= REJECTED_TICKER_ACRONYMS

    cashtags = [
        value for value in _EXPLICIT_TICKER_RE.findall(source)
        if value not in REJECTED_TICKER_ACRONYMS
    ]

    unmarked = _PAREN_TICKER_RE.findall(source) + _BARE_TICKER_RE.findall(source)
    validated = [value for value in unmarked if value in valid]

    return _unique(cashtags + validated)


def extract_marked_tickers(text: str | None, known_tickers: Iterable[str] | None = None) -> list[str]:
    """Tickers with an explicit, position-independent textual marker:
    a cashtag ($AAPL), a parenthetical annotation ("Block Inc. (SQ)"), an
    exchange-prefixed mention ("NYSE: SQ", "Nasdaq: NVDA"), or an explicit
    phrase ("trades under ASST", "ticker symbol NVDA").

    This is deliberately narrower than extract_tickers(): it excludes bare,
    unmarked mentions (a stray "SQ" floating in running text isn't strong
    evidence on its own), keeping only forms where someone explicitly wrote
    a ticker as a ticker. Used by TickerResolver as evidence that survives
    regardless of where in the article it appears -- unlike a bare company
    name mention, which only counts as strong evidence in the primary
    (title/summary/lede) zone. See ArticleAnalyzer.analyze().
    """
    source = text or ""
    valid = KNOWN_TICKERS | {t.strip().upper() for t in (known_tickers or ()) if t}
    valid -= REJECTED_TICKER_ACRONYMS

    cashtags = [v for v in _EXPLICIT_TICKER_RE.findall(source) if v not in REJECTED_TICKER_ACRONYMS]
    parens = [v for v in _PAREN_TICKER_RE.findall(source) if v in valid]
    exchange = [v for v in _EXCHANGE_TICKER_RE.findall(source) if v in valid]
    trades_under = [v for v in _TRADES_UNDER_RE.findall(source) if v in valid]
    ticker_symbol = [v for v in _TICKER_SYMBOL_RE.findall(source) if v in valid]

    return _unique(cashtags + parens + exchange + trades_under + ticker_symbol)


def extract_companies(
    text: str | None,
    known_company_names: Iterable[str] | None = None,
    known_tickers: Iterable[str] | None = None,
) -> list[str]:
    """Extract company-name candidates.

    Three independent, structurally-justified sources of candidates:
      1. Legal-suffix anchored ("Apple Inc").
      2. Possessive-anchored ("Dell's earnings").
      3. Bare, unmarked mentions ("Waymo announced...") -- only accepted
         against a whitelist (the built-in KNOWN_BARE_COMPANY_NAMES plus
         any caller-supplied `known_company_names`, e.g. from a real
         ticker/company mapping). This is deliberately the *only* way a
         bare mention with no suffix/possessive marker is ever accepted --
         there is no "any capitalized phrase" fallback, since that's
         exactly the kind of noise this module used to produce.

    `known_tickers` is also accepted here (passed straight through to the
    shared validity check) so a bare acronym-shaped candidate is validated
    the same way extract_tickers() validates one.
    """
    source = text or ""
    known_names = KNOWN_BARE_COMPANY_NAMES | {n.strip() for n in (known_company_names or ()) if n}
    known_tick = KNOWN_TICKERS | {t.strip().upper() for t in (known_tickers or ()) if t}

    raw_candidates = _COMPANY_SUFFIX_RE.findall(source) + _COMPANY_POSSESSIVE_RE.findall(source)

    for match in _BARE_PHRASE_RE.finditer(source):
        phrase = match.group(0)
        # Note: the ticker-shaped branch below intentionally checks only
        # TICKER_DOUBLES_AS_COMPANY_NAME (a small subset of KNOWN_TICKERS),
        # not the full known_tick (every ticker from a caller-supplied
        # mapping, or even the full built-in KNOWN_TICKERS). Most tickers
        # are never spoken as the company's name in prose -- nobody calls
        # ExxonMobil "XOM" -- so accepting every known ticker here would put
        # a bare ticker symbol into the `companies` column, which is for
        # company *names*, not ticker symbols (extract_tickers() already
        # covers ticker recognition separately).
        if phrase in known_names or phrase.upper() in TICKER_DOUBLES_AS_COMPANY_NAME:
            raw_candidates.append(phrase)
            continue
        # Try all contiguous sub-phrases so a known company can be found
        # anywhere inside a longer capitalized phrase.
        words = phrase.split()

        for start in range(len(words)):
            for end in range(len(words), start, -1):
                sub = " ".join(words[start:end])

                if (
                    sub in known_names
                    or sub.upper() in TICKER_DOUBLES_AS_COMPANY_NAME
                ):
                    raw_candidates.append(sub)
                    break

    normalized = [normalize_company_name(candidate) for candidate in raw_candidates]
    valid = [name for name in normalized if is_valid_company_name(name, known_tickers=known_tick)]
    return _unique(_drop_shorter_overlaps(valid))


def extract_events(text: str | None) -> list[str]:
    source = text or ""
    lower = source.lower()
    lede = lower[:_EVENT_LEDE_WINDOW]
    events = []
    for term in _EVENT_TERMS:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        matches = list(re.finditer(pattern, lower))
        if not matches:
            continue
        # A term in the lede, or mentioned more than once anywhere, is
        # treated as a real event rather than a single passing reference
        # buried deep in the article.
        if len(matches) > 1 or re.search(pattern, lede):
            events.append(term)
    return events


def extract_financial_entities(
    text: str | None,
    known_tickers: Iterable[str] | None = None,
    known_company_names: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    return FinancialEntities(
        companies=extract_companies(text, known_company_names=known_company_names, known_tickers=known_tickers),
        tickers=extract_tickers(text, known_tickers=known_tickers),
        percentages=extract_percentages(text),
        currency_values=extract_currency_values(text),
        events=extract_events(text),
    ).to_dict()
