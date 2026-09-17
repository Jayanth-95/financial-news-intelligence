import feedparser

# Common Reuters RSS feeds; start with business and world/top news
FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.reuters.com/reuters/technologyNews",
]


def fetch_items(days: int = 1):
    for url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            yield {
                "source": "reuters",
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", ""),
                "url": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
            }
