"""Publication-date parsing and freshness-window filtering.

Supports the date formats the shipped source adapters actually produce:
  * RFC 2822 (feedparser's `entry.published`, from RSS <pubDate>), e.g.
    "Tue, 23 Apr 2024 10:15:00 +0530"
  * ISO 8601 / RFC 3339 (sources/reddit.py's own formatting), e.g.
    "2024-04-23T10:15:00Z" or "2024-04-23T10:15:00+05:30"
  * A bare date, e.g. "2024-04-23"

A naive (timezone-less) timestamp is assumed to be UTC, since that's the
most defensible default for a source that didn't say otherwise -- silently
guessing a specific local timezone would be worse than a documented UTC
assumption.

Missing or unparseable dates are documented, not silently dropped -- see
is_within_window()'s docstring.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

LOG = logging.getLogger(__name__)


def parse_published(value: str | None) -> datetime | None:
    """Parse a publication timestamp into an aware UTC datetime, or None.

    Returns None for missing/empty/unparseable input rather than raising --
    callers decide how to treat that (see is_within_window).
    """
    if not value or not value.strip():
        return None
    text = value.strip()

    # RFC 2822 (typical RSS <pubDate>, what feedparser hands back verbatim).
    try:
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return _to_utc(parsed)
    except (TypeError, ValueError):
        pass

    # ISO 8601 / RFC 3339, including a trailing "Z" (Python's fromisoformat
    # didn't accept a bare "Z" until 3.11; handle it explicitly for 3.10-).
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return _to_utc(datetime.fromisoformat(iso_text))
    except ValueError:
        pass

    LOG.debug("Could not parse publication date: %r", value)
    return None


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        # No timezone was provided -- UTC is assumed rather than guessing
        # at a specific local timezone the source never stated.
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def is_within_window(published: str | None, days: int, *, now: datetime | None = None) -> bool:
    """True if `published` falls within the last `days` days of `now`.

    Documented behavior for missing/unparseable dates: they are accepted
    (fail open), not rejected. Several source adapters -- particularly the
    HTML-crawl fallback paths in cnbc/investing/marketwatch/nasdaq -- don't
    always have a reliable published field to report, and rejecting
    everything without one would silently drop legitimate current content
    just because a source's date field is empty. This is a deliberate
    trade-off, not a silent gap: an article with a *known*, parseable date
    outside the window is always rejected regardless of this default; only
    genuinely unknown dates get the benefit of the doubt.
    """
    if days is None or days <= 0:
        return True
    parsed = parse_published(published)
    if parsed is None:
        return True
    reference = now if now is not None else datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    cutoff = reference - timedelta(days=days)
    return parsed >= cutoff
