from utils.network import TimeoutFeedParser, fetch_with_timeout
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# prefer RSS feeds if available; allow homepage as fallback
FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC top news RSS
    "https://www.cnbc.com/world/?region=world",
]


def _crawl_links(page_url: str, max_links: int = 30):
    content = fetch_with_timeout(page_url)
    if not content:
        return
    soup = BeautifulSoup(content, "html.parser")
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # normalize
        if href.startswith("//"):
            href = urlparse(page_url).scheme + ":" + href
        href = urljoin(page_url, href)
        if href in seen:
            continue
        seen.add(href)
        # basic heuristics for article URLs
        if "/news/" in href or "/article/" in href or "/videos/" in href or "/stories/" in href:
            yield {"source": "cnbc", "title": a.get_text(strip=True) or href, "summary": "", "url": href, "published": ""}
        if len(seen) >= max_links:
            break


def fetch_items(days: int = 1):
    parser = TimeoutFeedParser()
    for url in FEEDS:
        feed = parser.parse(url)
        if getattr(feed, "entries", None):
            for e in feed.entries:
                yield {
                    "source": "cnbc",
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", ""),
                    "url": getattr(e, "link", ""),
                    "published": getattr(e, "published", ""),
                }
        else:
            # fallback to light crawl
            for item in _crawl_links(url):
                yield item
