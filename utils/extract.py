"""Article text extraction and normalization."""

from __future__ import annotations
import re
from typing import Any
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:
    trafilatura = None

_WHITESPACE_RE = re.compile(r"\s+")

def clean_text(value: Any) -> str:
    """Normalize arbitrary text into compact whitespace-separated text."""
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()

def extract_main_text(html: str | bytes | None) -> str:
    """Extract article text with Trafilatura and a BeautifulSoup fallback."""
    if not html:
        return ""
    if trafilatura is not None:
        try:
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if text:
                return clean_text(text)
        except Exception:
            pass
    try:
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "noscript", "template"]):
            element.decompose()
        return clean_text(soup.get_text(" "))
    except Exception:
        return ""

def extract_article_text(html: str | bytes | None) -> str:
    """Alias for callers using article-oriented terminology."""
    return extract_main_text(html)
