
import feedparser

# Use FT RSS summaries only (do not bypass paywall).
FEEDS = [
    "https://www.ft.com/rss/companies",  # Companies section
    "https://www.ft.com/rss/markets",    # Markets section
]

def fetch_items(days: int = 1):
    for url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            yield {
                "source": "financial_times_rss",
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", ""),
                "url": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
            }
