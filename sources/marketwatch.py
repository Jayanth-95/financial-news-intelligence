from utils.network import TimeoutFeedParser, fetch_with_timeout
from bs4 import BeautifulSoup
from urllib.parse import urljoin

FEEDS = [
    "https://www.marketwatch.com/",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
]


def _crawl_links(page_url: str, max_links: int = 40):
    content = fetch_with_timeout(page_url)
    if not content:
        return
    soup = BeautifulSoup(content, "html.parser")
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        href = urljoin(page_url, href)
        if href in seen:
            continue
        seen.add(href)
        if "/story/" in href or "/news/" in href or "/investing/" in href:
            yield {"source": "marketwatch", "title": a.get_text(strip=True) or href, "summary": "", "url": href, "published": ""}
        if len(seen) >= max_links:
            break


def fetch_items(days: int = 1):
    parser = TimeoutFeedParser()
    for url in FEEDS:
        feed = parser.parse(url)
        if getattr(feed, "entries", None):
            for e in feed.entries:
                yield {
                    "source": "marketwatch",
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", ""),
                    "url": getattr(e, "link", ""),
                    "published": getattr(e, "published", ""),
                }
        else:
            for item in _crawl_links(url):
                yield item
