
import feedparser

# Moneycontrol finance-related RSS feeds (public). You can add more.
FEEDS = [
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.moneycontrol.com/rss/marketreports.xml",
]

def fetch_items(days: int = 1):
    for url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            yield {
                "source": "moneycontrol",
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", ""),
                "url": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
            }
