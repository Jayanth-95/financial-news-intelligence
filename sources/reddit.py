from __future__ import annotations
import os
from typing import List
from datetime import datetime

from utils.network import get_session

# Subreddits to poll for market-related discussion
SUBREDDITS: List[str] = [
    "NASDAQ",
    "stocks",
    "investing",
    "StockMarket",
    "wallstreetbets",
]


def _item_from_submission(submission, subreddit: str):
    title = getattr(submission, "title", "")
    selftext = getattr(submission, "selftext", "") or ""
    link = getattr(submission, "url", "")
    created = getattr(submission, "created_utc", None)
    if created:
        try:
            published = datetime.utcfromtimestamp(float(created)).isoformat() + "Z"
        except Exception:
            published = ""
    else:
        published = ""

    return {
        "source": f"reddit/r/{subreddit}",
        "title": title,
        "summary": (selftext[:1000] if selftext else ""),
        "url": link,
        "published": published,
    }


def fetch_items(days: int = 1):
    """Fetch recent posts. Prefer using PRAW with credentials from environment
    (.env) if available; otherwise fall back to Reddit's public JSON endpoints.
    """
    # Try to use PRAW (recommended). Credentials should be in environment:
    # REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    try:
        import praw
    except Exception:
        praw = None

    if praw and os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET"):
        # Use authenticated PRAW client (read-only by default)
        try:
            reddit = praw.Reddit(
                client_id=os.environ.get("REDDIT_CLIENT_ID"),
                client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
                user_agent=os.environ.get("REDDIT_USER_AGENT", "news-sentiment-bot/1.0"),
            )
            # iterate configured subreddits
            for sub in SUBREDDITS:
                try:
                    for submission in reddit.subreddit(sub).new(limit=100):
                        yield _item_from_submission(submission, sub)
                except Exception:
                    continue
            return
        except Exception:
            # fall through to unauthenticated JSON fallback
            pass

    # Fallback: unauthenticated public JSON endpoints (rate-limited)
    sess = get_session()
    headers = {"User-Agent": os.environ.get("REDDIT_USER_AGENT", "news-sentiment-bot/1.0")}
    for sub in SUBREDDITS:
        url = f"https://www.reddit.com/r/{sub}/new.json?limit=100"
        try:
            r = sess.get(url, timeout=10, headers=headers)
            r.raise_for_status()
            j = r.json()
        except Exception:
            continue
        posts = j.get("data", {}).get("children", [])
        for p in posts:
            try:
                d = p.get("data", {})
                title = d.get("title", "")
                selftext = d.get("selftext", "") or ""
                link = d.get("url", "")
                created = d.get("created_utc")
                if created:
                    try:
                        published = datetime.utcfromtimestamp(float(created)).isoformat() + "Z"
                    except Exception:
                        published = ""
                else:
                    published = ""
                yield {
                    "source": f"reddit/r/{sub}",
                    "title": title,
                    "summary": (selftext[:1000] if selftext else ""),
                    "url": link,
                    "published": published,
                }
            except Exception:
                continue
