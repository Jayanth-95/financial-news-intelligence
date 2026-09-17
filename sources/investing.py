from utils.network import TimeoutFeedParser, fetch_with_timeout
from bs4 import BeautifulSoup
from urllib.parse import urljoin

FEEDS = [
    "https://www.investing.com/news/",
    "https://www.investing.com/rss/news.rss",
]


def _crawl_links(page_url: str, max_links: int = 50):
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
        if "/news/" in href or "/article/" in href:
            yield {"source": "investing", "title": a.get_text(strip=True) or href, "summary": "", "url": href, "published": ""}
        if len(seen) >= max_links:
            break


def fetch_items(days: int = 1):
    parser = TimeoutFeedParser()
    for url in FEEDS:
        feed = parser.parse(url)
        if getattr(feed, "entries", None):
            for e in feed.entries:
                yield {
                    "source": "investing",
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", ""),
                    "url": getattr(e, "link", ""),
                    "published": getattr(e, "published", ""),
                }
        else:
            for item in _crawl_links(url):
                yield item
