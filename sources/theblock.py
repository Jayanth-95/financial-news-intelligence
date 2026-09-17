
import feedparser

FEEDS = [
    "https://www.theblock.co/rss.xml",
]

def fetch_items(days: int = 1):
    for url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            yield {
                "source": "theblock",
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", ""),
                "url": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
            }
