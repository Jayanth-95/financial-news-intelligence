"""Shared HTTP and RSS networking utilities.

The rest of the application should use this module for outbound HTTP calls.
That keeps timeout, retry, headers, and error handling consistent across all
news-source integrations.
"""

from __future__ import annotations

import logging
from typing import Optional

import feedparser
import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import TIMEOUT, USER_AGENT

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = TIMEOUT
MAX_RETRIES = 3
_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
_RETRY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class NetworkClient:
    """Small HTTP client with application-wide networking policy."""

    def __init__(
        self,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self._session = self._build_session(max_retries)

    @staticmethod
    def _build_session(max_retries: int) -> Session:
        retry_policy = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=0.5,
            status_forcelist=_RETRY_STATUS_CODES,
            allowed_methods=_RETRY_METHODS,
            respect_retry_after_header=True,
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_policy, pool_connections=10, pool_maxsize=20)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": USER_AGENT})
        return session

    def get(self, url: str, *, timeout: Optional[int] = None) -> Response:
        """Perform a GET using the shared network policy."""
        return self._session.get(
            url,
            timeout=timeout or self.timeout,
            headers={"User-Agent": self.user_agent},
        )

    def close(self) -> None:
        self._session.close()


_CLIENT = NetworkClient()


def get_session() -> Session:
    """Return the shared requests session used by source integrations.

    Kept as a compatibility API because Reddit and a few source adapters use
    requests directly. The session owns connection pooling and retry policy.
    """
    return _CLIENT._session


def fetch_with_timeout(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[bytes]:
    """Fetch URL content and return bytes, or ``None`` on a network failure."""
    try:
        response = _CLIENT.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        logger.warning("HTTP request failed for %s: %s", url, exc)
        return None


class TimeoutFeedParser:
    """Fetch and parse an RSS/Atom feed without mutating global socket state."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def parse(self, url: str):
        """Return a ``feedparser`` result for ``url``.

        The previous implementation changed ``socket.setdefaulttimeout`` for
        the entire Python process. Fetching the feed through our HTTP client
        gives each request an isolated timeout and avoids global side effects.
        """
        content = fetch_with_timeout(url, timeout=self.timeout)
        if content is None:
            return feedparser.FeedParserDict()

        try:
            return feedparser.parse(content)
        except (ValueError, TypeError) as exc:
            logger.warning("RSS parsing failed for %s: %s", url, exc)
            return feedparser.FeedParserDict()
