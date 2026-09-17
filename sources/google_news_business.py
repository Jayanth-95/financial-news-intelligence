
import feedparser

# Proxy for Google Finance: use Google News - Business RSS (official public feed).
FEEDS = [
    "https://news.google.com/news/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
]

def fetch_items(days: int = 1):
    for url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            yield {
                "source": "google_news_business",
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", ""),
                "url": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
            }
