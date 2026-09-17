"""Deterministic offline smoke test for the financial-news pipeline."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.runner import PipelineRunner
from app.services import (
    AnalysisOptions,
    ArticleAnalyzer,
    ArticleExtractor,
    SentimentService,
    TickerResolver,
)


class OfflineExtractor(ArticleExtractor):
    """Use supplied article metadata instead of making network requests."""

    def extract(self, item):
        return f"{item.title}. {item.summary}"


class SampleSource:
    """Small deterministic source used only for local validation."""

    def __call__(self, days=1):
        # Published relative to "now" (not a hardcoded calendar date), so
        # this fixture stays inside the pipeline's real --days publication
        # window (see utils/dates.py) no matter when the smoke test runs.
        now = datetime.now(timezone.utc)
        yield {
            "source": "sample",
            "title": "Example Corp reports strong earnings growth",
            "summary": "Revenue increased 12% and profit reached $250 million.",
            "url": "",
            "published": (now - timedelta(hours=2)).isoformat(),
        }
        yield {
            "source": "sample",
            "title": "Example Markets announces dividend increase",
            "summary": "The company announced a 5% dividend increase.",
            "url": "",
            "published": (now - timedelta(hours=1)).isoformat(),
        }


def main():
    output = Path("data/sample_output.csv")
    analyzer = ArticleAnalyzer(
        OfflineExtractor(),
        TickerResolver({}),
        SentimentService(enabled=False),
        AnalysisOptions(use_spacy=False),
    )
    count = PipelineRunner([SampleSource()], analyzer, output, days=1).execute()
    print(f"Offline smoke test passed: {count} rows written to {output}")


if __name__ == "__main__":
    main()
