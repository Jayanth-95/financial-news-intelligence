"""Deterministic financial-news relevance filtering."""

from __future__ import annotations
import re
from typing import Iterable

FINANCIAL_TERMS = (
    "earnings", "guidance", "ipo", "dividend", "buyback", "upgrade",
    "downgrade", "shares", "stock", "stocks", "market", "markets",
    "index", "indices", "results", "profit", "profits", "loss", "losses",
    "revenue", "forecast", "merger", "acquisition", "layoff", "layoffs",
    "nasdaq", "nyse", "nse", "bse",    "hedge",
    "hedging",
    "fund",
    "funds",
    "portfolio",
    "investment",
    "investments",
    "investor",
    "investors",
    "trading",
    "trader",
    "traders",
    "stake",
    "holding",
    "holdings",
    "capital",
    "asset",
    "assets",
    "wealth",
    "private equity",
    "venture capital", "sensex", "nifty", "csi 300",
)

_FINANCIAL_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(t) for t in FINANCIAL_TERMS) + r")(?!\w)",
    re.IGNORECASE,
)

def is_market_relevant(text: str | None) -> bool:
    """Return True when text contains a financial-market term."""
    return bool(text and _FINANCIAL_PATTERN.search(text))

def is_stock_market_related(text: str | None) -> bool:
    """Backward-compatible name used by the existing pipeline."""
    return is_market_relevant(text)

def contains_financial_terms(text: str | None, terms: Iterable[str] | None = None) -> bool:
    """Check text against the default or a caller-supplied vocabulary."""
    if not text:
        return False
    if terms is None:
        return is_market_relevant(text)
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)",
        re.IGNORECASE,
    )
    return bool(pattern.search(text))
