"""Shared structural rules and small curated domain-knowledge sets for
company/organization/ticker candidates.

utils/parse.py (deterministic regex) and utils/ner.py (spaCy) each produce
their own list of candidates from the same article text. This module is
the single place both agree on: what counts as a generic/non-name term,
what a closed-class leading word looks like, which short all-caps tokens
are recognized tickers vs. common acronyms, and how a possessive is
normalized. Keeping this in one place -- instead of independently
maintained, drifting rule sets -- is what keeps the two extractors
consistent.
"""
from __future__ import annotations

import re

# --- Generic / boilerplate / closed grammatical classes -------------------

# Generic organizational/financial nouns that are never themselves a company
# name, regardless of which extractor produced the candidate. This also
# covers bare legal-entity suffixes (e.g. "Holdings", "Corp") appearing with
# no attached name -- those are structurally meaningless on their own.
# This is a small, closed set of generic nouns, not an attempt to enumerate
# every possible source of noise.
GENERIC_ORG_TERMS = frozenset({
    "company", "companies", "market", "markets", "stock market", "wall street",
    "government", "investors", "investor", "financial services", "industry",
    "industries", "bank", "banks", "club",
    "inc", "corp", "corporation", "ltd", "limited", "plc", "holdings",
    "group", "technologies", "technology",
})

# Boilerplate/navigation phrases that occasionally get tagged as an "ORG"
# by a generic NER model when they slip past the article boilerplate filter.
BOILERPLATE_TERMS = frozenset({
    "register", "get", "follow", "access", "read", "subscribe", "subscription",
    "sign in", "log in", "login", "newsletter", "newsletters", "cookie",
    "privacy", "terms", "conditions", "terms and conditions", "terms of service",
    "terms of use", "alphaville", "explore more offers", "see",
})

# Closed-class function words (determiners, conjunctions, common adverbs)
# that must never be treated as the start of a proper-noun company name.
# This is a fixed grammatical category -- not a domain-specific noise list --
# so it stays small no matter how much financial vocabulary changes.
LEADING_STOPWORDS = frozenset({
    "The", "A", "An", "This", "That", "These", "Those", "Also", "And", "But",
    "Or", "Some", "Many", "Other", "Its", "Their", "His", "Her", "Our", "Your",
    "My", "In", "On", "At", "By", "For", "With", "From", "As", "Is", "Are",
    "Was", "Were", "It", "While", "When", "If", "Because", "Although",
    "Since", "Unless", "After", "Before", "Until", "Once", "Whereas", "Last", "Next",
    "Following", "During", "Amid", "Despite", "Regarding", "Concerning",
})

# First/second/third-person pronouns mistagged as ORG by a generic NER
# model (e.g. "She" appearing bare in a byline-adjacent sentence). A fixed,
# closed grammatical category -- not a name list.
PRONOUNS = frozenset({
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "theirs",
    "i", "we", "you", "himself", "herself", "itself", "themselves",
})

# Organization-type words that, as the LAST word of a multi-word candidate,
# indicate a non-company institution (think tank, government body,
# university, etc.) rather than a market-relevant company. A small, closed
# set of institution-type nouns -- not an attempt to enumerate every real
# institution name (that would be a blacklist; this rejects by structural
# word-ending pattern instead, the same approach already used for legal
# entity suffixes in GENERIC_ORG_TERMS).
NON_COMPANY_ORG_SUFFIXES = frozenset({
    "institution", "foundation", "university", "council", "committee",
    "department", "ministry", "agency", "administration", "commission",
    "bureau", "reserve", "party", "college", "academy", "trust", "club",
    "school",
})

# Days of the week, month names, and common temporal/spatial deictic words.
# A fixed, closed linguistic category (not domain noise): a generic NER
# model sometimes mistags these as ORG in headline-style text ("...report
# earnings Wednesday", "...cut rates in February"), and no amount of
# financial-vocabulary drift changes what a weekday or month name is.
# Reused by both the regex and spaCy paths.
TEMPORAL_DEICTIC_TERMS = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "yesterday", "tomorrow", "tonight", "here", "there", "now", "then",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
})

# Government/political bodies and figures-by-title -- a market-relevant
# "company" extractor should not treat these as companies. Small, closed,
# and disclosed as incomplete: this covers commonly-recurring bodies in
# financial/political news coverage, not every government entity that
# could ever be mentioned.
GOVERNMENT_POLITICAL_TERMS = frozenset({
    "fed", "fedwatch", "federal reserve", "white house", "state of the union",
    "house", "senate", "congress", "ninth circuit", "supreme court",
    "treasury", "pentagon", "capitol", "parliament",
})

# Stock indexes and exchanges: real, trackable market benchmarks, but not
# themselves companies with a ticker to resolve. Small, closed, disclosed
# as incomplete -- covers the handful of major indexes/exchanges that
# recur constantly in financial news.
MARKET_INFRASTRUCTURE_TERMS = frozenset({
    "s&p 500", "s&p", "nasdaq composite", "nasdaq", "dow jones",
    "dow jones industrial average", "russell 2000", "ftse 100", "nikkei 225",
    "sensex", "nifty 50", "nifty", "new york stock exchange",
    "nasdaq stock market", "london stock exchange", "bombay stock exchange",
    "national stock exchange",
})

# Major news outlets: a source reporting on itself within its own article
# ("CNBC TV", "CNBC.com", "CNBC Make It") is a self-reference, not a
# company being covered. Matched as a whole-phrase or leading-word prefix
# (see is_valid_company_name), so "CNBC X" is rejected generally rather
# than needing every specific CNBC property name enumerated.
MEDIA_ORGANIZATION_TERMS = frozenset({
    "cnbc", "reuters", "bloomberg", "marketwatch", "yahoo finance",
    "financial times", "wall street journal", "moneycontrol", "the block",
    "google news", "associated press", "business insider",
})

# Social-media products/platforms: often owned by a company relevant to
# financial news (Meta owns Instagram, Alphabet owns YouTube), but the
# platform name itself is a product, not the company whose stock is being
# discussed -- see is_valid_company_name's word-level check.
SOCIAL_MEDIA_PRODUCT_TERMS = frozenset({
    "instagram", "youtube", "tiktok", "facebook", "whatsapp", "linkedin",
    "snapchat", "pinterest",
})

# A small set of common role/title words that, immediately preceding a
# person's name, indicate the whole phrase ("Fed Chairman Kevin Warsh",
# "President Trump") is fundamentally a reference to that PERSON, not a
# company -- see is_valid_company_name's PERSON-suffix check in
# utils/ner.py. This is a closed grammatical/role category, not a list of
# names.
TITLE_PREFIX_WORDS = frozenset({
    "president", "chairman", "chairwoman", "chair", "ceo", "cfo", "coo", "cto",
    "founder", "director", "governor", "senator", "secretary", "minister",
    "chancellor", "mayor", "judge", "justice", "analyst", "economist",
    "spokesperson", "spokesman", "spokeswoman", "editor", "reporter", "anchor",
})

# Common US timezone abbreviations that occasionally get glued onto the
# front of an adjacent proper noun by a span-boundary error (e.g. a
# timestamp "4:00 PM ET" immediately followed by a company name, merged
# into one span: "ET CrowdStrike"). A small, closed, purely grammatical
# category, not a name list.
TIMEZONE_ABBREVIATIONS = frozenset({"et", "pt", "ct", "mt", "edt", "pdt", "cdt", "mdt", "est", "pst", "cst", "mst", "gmt", "utc"})

# Words that, as the LAST word of an otherwise-plausible multi-word
# candidate, are common company-DESCRIPTION adjectives rather than part of
# the name itself -- a span-boundary error that merges a company name with
# the adjective that describes it in running prose ("ExxonMobil ... is an
# integrated oil and gas company" -> "Exxon Mobil Integrated"). A small,
# closed set of descriptive adjectives, not company-specific.
DANGLING_DESCRIPTOR_WORDS = frozenset({
    "integrated", "diversified", "multinational", "global", "leading",
    "major", "largest", "publicly", "privately", "listed", "based",
})

# Bare prepositions that should never appear as a standalone word ANYWHERE
# within a company-name candidate. Real company names essentially never
# contain "on"/"for" as a freestanding word (unlike "of", deliberately
# excluded here to protect real names like "Bank of America"); a candidate
# containing one is almost always a scraped link/headline fragment ("Exxon
# Mobil Stock Buybacks on TipRanks", "Institute for Supply Management's...").
INTERNAL_DANGLING_PREPOSITIONS = frozenset({"on", "for"})

# Business-type words that, as the LAST word of the portion of a candidate
# BEFORE a trailing person-name match, signal that portion is already a
# complete, plausible company name with a person's name spuriously
# glued on after it ("Diamondback Energy Mehta" -- "Diamondback Energy" is
# already a complete name; "Mehta" is a stray analyst/exec surname). Used
# only by utils.ner's person-suffix check, alongside (not instead of)
# TITLE_PREFIX_WORDS -- this is what lets that check work even with no
# title word present.
BUSINESS_TYPE_SUFFIX_WORDS = frozenset({
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "plc",
    "co", "company", "holdings", "group", "technologies", "technology",
    "energy", "capital", "partners", "ventures", "networks", "systems",
    "solutions", "industries", "labs", "laboratories", "resources",
    "financial", "financials", "markets", "bank", "motors", "pharmaceuticals",
    "therapeutics", "biosciences", "software", "media", "communications",
    "airlines", "properties", "realty",
})

# --- Tickers ---------------------------------------------------------------
#
# A small, curated set of real, actively-traded ticker symbols. This is
# domain knowledge, not invention: every entry is a genuine symbol, included
# so well-known companies can be recognized in bare/parenthetical form even
# without a caller-supplied ticker map. It is deliberately small and
# non-exhaustive -- an unrecognized symbol is reported as unresolved, never
# guessed. Callers with a real ticker mapping should extend recognition via
# their own known_tickers, not by growing this list arbitrarily.
KNOWN_TICKERS = frozenset({
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD",
    "INTC", "IBM", "ORCL", "CRM", "ADBE", "PYPL", "DIS", "KO", "PEP", "WMT",
    "JPM", "GS", "MS", "BAC", "C", "V", "MA", "HD", "MCD", "NKE", "SBUX",
    "BA", "GE", "F", "GM", "XOM", "CVX", "PFE", "JNJ", "UNH", "T", "VZ",
    "CSCO", "QCOM", "TXN", "UBER", "ABNB", "SNAP", "SHOP", "SQ", "COIN",
    "PLTR", "RIVN", "LCID", "NIO", "BABA", "TSM", "CRWD",
})

# A much smaller subset of KNOWN_TICKERS: symbols that are ALSO commonly
# used as the company's colloquial name in ordinary prose (people say "AMD"
# or "IBM", not "Advanced Micro Devices" or "International Business
# Machines"). Used only by extract_companies()'s bare-mention path, to
# decide when a bare ticker-shaped token is itself a valid company-name
# candidate. Deliberately much narrower than KNOWN_TICKERS: most tickers
# (e.g. "XOM", "NVDA", "CRWD") are never spoken as the company's name --
# people say "Exxon", "Nvidia", "CrowdStrike" -- so accepting every known
# ticker here would put a bare ticker symbol into the `companies` column,
# which is for company names, not ticker symbols.
TICKER_DOUBLES_AS_COMPANY_NAME = frozenset({"AMD", "IBM", "GE", "GM", "F", "T", "V", "C"})

# Common financial/business acronyms that are short and all-caps -- exactly
# what a ticker (or a bare company mention) looks like -- but are not. This
# is a defensive backstop, not the primary defense (default-deny already
# rejects anything not in KNOWN_TICKERS). Entries here take precedence even
# over a caller-supplied known_tickers set, since these acronyms appear
# constantly in ordinary financial writing. ("AI" is technically also
# C3.ai's real ticker, but the acronym usage overwhelmingly dominates in
# financial news, so it's deliberately excluded here in favor of precision.)
REJECTED_TICKER_ACRONYMS = frozenset({
    "EPS", "IPO", "CEO", "CFO", "COO", "CTO", "GDP", "YOY", "QOQ", "MOM",
    "ETF", "SEC", "ESG", "EV", "FY", "AI", "API", "CSR", "KPI", "PE", "ROI",
    "CAGR", "EBIT", "NASDAQ", "NYSE", "CNBC",
})

# --- Bare company names -----------------------------------------------------
#
# Real companies commonly referenced by name alone, with no legal suffix
# ("Inc"/"Corp"/...) and no possessive marker -- "Waymo announced...", not
# "Waymo Inc" or "Waymo's". A pure regex has no way to know these are
# companies rather than any other capitalized word (that's what a real NER
# model is for -- see utils/ner.py); this whitelist is what lets the
# deterministic regex path *also* recognize a small set of well-known bare
# names without depending on spaCy being installed/available. Deliberately
# small, curated, and disclosed as incomplete: an unlisted bare company name
# is not extracted by regex, and depends on the spaCy layer instead, exactly
# like the rest of this file's default-deny design.
KNOWN_BARE_COMPANY_NAMES = frozenset({
    "Google", "Alphabet", "Waymo", "Amazon", "Meta", "Apple", "Microsoft",
    "Tesla", "Nvidia", "Intel", "Salesforce", "CrowdStrike", "Oracle",
    "Adobe", "Cisco", "Qualcomm", "PayPal", "Disney", "Nike", "Starbucks",
    "Boeing", "Walmart", "Netflix", "Uber", "Airbnb", "Spotify", "Shopify",
    "Reliance", "Infosys", "Wipro", "Zomato", "Paytm", "Flipkart", "AT&T",
    "Snowflake", "Palantir", "Coinbase", "Robinhood", "Moderna", "Pfizer",
    "O'Reilly", "O'Reilly Automotive",
    "Bodycote", "Veritas", "Shein", "Solana", "ServiceTitan", "Cytokinetics",
    "Alkermes", "Strive", "Citi", "Anthropic", "Lululemon", "Multicoin Capital",
    "Grayscale", "Fastly", "PubMatic", "Gevo",
    # Brand names whose real legal name genuinely ends in an apostrophe-s
    # (see PRESERVED_POSSESSIVE_NAMES below) -- listed here too so the
    # bare-mention regex path recognizes them in the first place.
    "McDonald's", "Levi's", "Kohl's", "Macy's", "Lowe's", "Wendy's",
    "Denny's", "Arby's", "Chili's", "Sam's Club", "Trader Joe's",
    "Domino's","Berkshire Hathaway",
"Goldman Sachs",
"Goldman",
"Novo Nordisk",
"Frasers Group",
"Michael Dell", "Culver's", "Dick's Sporting Goods", "Applebee's",
})

# Company names that end with an apostrophe-s as part of their actual
# legal/brand name, not a grammatical possessive -- normalize_company_name
# must NOT strip these. Small, closed, and disclosed as non-exhaustive; an
# unlisted apostrophe-s company name is treated the way English grammar
# normally would be (its possessive form is stripped), the same as any
# other company mentioned possessively (e.g. "Dell's earnings" -> "Dell").
PRESERVED_POSSESSIVE_NAMES = frozenset({
    "McDonald's", "Levi's", "Kohl's", "Macy's", "Lowe's", "Wendy's",
    "Denny's", "Arby's", "Chili's", "Sam's Club", "Trader Joe's",
    "Domino's", "Culver's", "Applebee's", "Dick's Sporting Goods",
})

_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_POSSESSIVE_RE = re.compile(r"[\u2019']s\Z")
_TRAILING_TICKER_ANNOTATION_RE = re.compile(r"^(.*\S)\s*\([A-Z]{1,6}\)\s*$")

# Legal-entity suffix words stripped only for TICKER-RESOLUTION comparison
# purposes (see app.services.TickerResolver._resolution_key) -- never for
# the extracted `companies` output itself, which keeps the suffix the
# extractor actually found. This is what lets "Berkshire Hathaway Inc" (as
# extracted) match a mapping keyed just "Berkshire Hathaway" (or vice
# versa) without needing every suffix variant listed in the mapping file.
RESOLUTION_SUFFIX_WORDS = frozenset({
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "plc",
    "co", "company", "holdings", "group",
})


def normalize_company_name(value: str | None) -> str:
    """Collapse whitespace, trim punctuation, and strip a trailing possessive
    or a trailing "(TICKER)" annotation.

    "Dell's" -> "Dell", "Lenovo's" -> "Lenovo", "HDFC Bank's" -> "HDFC Bank",
    "Apple Inc's" -> "Apple Inc". Only a *trailing* possessive marker is
    stripped -- an internal apostrophe (e.g. "O'Reilly") is untouched, and a
    name in PRESERVED_POSSESSIVE_NAMES (e.g. "McDonald's", "Dick's Sporting
    Goods") is returned exactly as-is, since there the apostrophe-s is part
    of the real name, not a grammatical possessive.

    "Nio (NIO)" -> "Nio": financial news very commonly annotates a company
    name with its ticker in parentheses; the ticker itself is separately
    handled by extract_tickers(), so it's stripped here rather than left in
    the displayed company name.
    """
    text = _WHITESPACE_RE.sub(" ", value or "").strip(" \t\r\n.,;:|\u2014\u2013-")
    if text in PRESERVED_POSSESSIVE_NAMES:
        return text
    text = _TRAILING_POSSESSIVE_RE.sub("", text)
    match = _TRAILING_TICKER_ANNOTATION_RE.match(text)
    if match:
        text = match.group(1)
    return text.strip()


def resolution_key(value: str | None) -> str:
    """Casefolded comparison key for ticker resolution, with a trailing
    legal-entity suffix stripped in addition to normalize_company_name's
    possessive-stripping. Used only for matching a company candidate against
    a ticker mapping -- never for the displayed `companies` column, which
    always keeps the extractor's original text.

    Each word has leading/trailing comma/period stripped before comparison
    -- this is what makes "Berkshire Hathaway, Inc." (the common
    comma-before-suffix convention in US corporate naming) match the same
    key as "Berkshire Hathaway Inc" or "Berkshire Hathaway": without this,
    the comma stays attached to "Hathaway," after "Inc." is popped, and the
    two variants never compare equal.
    """
    normalized = normalize_company_name(value)
    words = [w.strip(",.") for w in normalized.split()]
    words = [w for w in words if w]
    while words and words[-1].casefold() in RESOLUTION_SUFFIX_WORDS:
        words.pop()
    return " ".join(words).casefold()


def _looks_like_bare_uppercase_token(value: str) -> bool:
    """A single, all-uppercase, letters-only word -- exactly the shape of a
    ticker, a common acronym, or ALL-CAPS boilerplate text (e.g. leaked
    "TERMS AND CONDITIONS" footer fragments), and structurally
    indistinguishable from a real ticker without a whitelist. No length cap:
    a genuine company name is essentially never written in full uppercase in
    ordinary prose (only acronyms/tickers or shouted boilerplate are), so
    this applies regardless of how long the all-caps word is.
    """
    return " " not in value and value.isalpha() and value.isupper()


def _normalize_for_set_lookup(value: str) -> str:
    """Casefold and collapse spacing around "&" (e.g. "S & P" -> "s&p") so
    set membership checks aren't defeated by incidental whitespace variants
    of the same compound name.
    """
    folded = value.casefold()
    return re.sub(r"\s*&\s*", "&", folded)


def is_valid_company_name(
    value: str, *, max_words: int = 8, known_tickers: "frozenset[str] | set[str] | None" = None
) -> bool:
    """Structural validity check shared by every company/ORG candidate source.

    `known_tickers` (defaulting to the built-in KNOWN_TICKERS) governs one
    specific case: a bare, all-uppercase single token (e.g. "AI", "CNBC",
    "CONDITIONS", "IBM") is only accepted if it's a recognized ticker -- the
    same default-deny approach used for ticker extraction, applied here so a
    generic NER model's all-caps false positives get the same treatment as
    extract_tickers()'s.

    Beyond generic/temporal/institution-suffix checks, this also rejects
    (all via small, closed, disclosed term sets -- see their definitions
    above): government/political bodies, stock indexes and exchanges, news
    outlets self-referencing within their own articles, social-media
    product/platform names, pronouns, and garbled single-word fusion
    artifacts.
    """
    if not value or len(value) < 2:
        return False
    if value.count("(") != value.count(")"):
        return False
    if value in LEADING_STOPWORDS:
        return False
    folded = value.casefold()
    if folded in BOILERPLATE_TERMS or folded in GENERIC_ORG_TERMS:
        return False
    if folded in TEMPORAL_DEICTIC_TERMS or folded in PRONOUNS:
        return False
    if any(folded.startswith(prefix) for prefix in ("register ", "access ", "follow ", "read free", "see ")):
        return False

    words = value.split()
    if len(words) > max_words:
        return False
    if words[0].casefold() in {w.casefold() for w in LEADING_STOPWORDS}:
        return False
    if words[0].casefold() in TIMEZONE_ABBREVIATIONS:
        return False
    if words[-1].casefold() in TEMPORAL_DEICTIC_TERMS:
        return False
    if len(words) > 1 and words[-1].casefold() in NON_COMPANY_ORG_SUFFIXES:
        return False
    if len(words) > 1 and words[-1].casefold() in DANGLING_DESCRIPTOR_WORDS:
        return False
    if any(word.casefold() in SOCIAL_MEDIA_PRODUCT_TERMS for word in words):
        return False
    if any(word.casefold() in INTERNAL_DANGLING_PREPOSITIONS for word in words):
        return False

    lookup_key = _normalize_for_set_lookup(value)
    if lookup_key in GOVERNMENT_POLITICAL_TERMS or lookup_key in MARKET_INFRASTRUCTURE_TERMS:
        return False
    if any(
        lookup_key == outlet or lookup_key.startswith(outlet + " ") or lookup_key.startswith(outlet + ".")
        for outlet in MEDIA_ORGANIZATION_TERMS
    ):
        return False

    if _looks_like_bare_uppercase_token(value):
        allowed = known_tickers if known_tickers is not None else KNOWN_TICKERS
        if value.upper() not in allowed or value.upper() in REJECTED_TICKER_ACRONYMS:
            return False
    return True
