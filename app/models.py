from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class NewsItem:
    source: str
    title: str
    summary: str = ""
    url: str = ""
    published: str = ""
    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "NewsItem":
        return cls(*(str(value.get(k, "") or "").strip() for k in ("source","title","summary","url","published")))

@dataclass(frozen=True)
class Sentiment:
    label: str
    score: float

@dataclass
class ArticleAnalysis:
    item: NewsItem
    text: str
    companies: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    mapped_tickers: list[str] = field(default_factory=list)
    percentages: list[str] = field(default_factory=list)
    currency_values: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    sentiment: Sentiment = field(default_factory=lambda: Sentiment("skipped",0.0))
    def as_row(self):
        return {"source":self.item.source,"url":self.item.url,"published":self.item.published,"title":self.item.title,"summary":self.item.summary,"text":self.text[:4000],"companies":", ".join(self.companies),"tickers":", ".join(self.tickers),"mapped_tickers":", ".join(self.mapped_tickers),"percentages":", ".join(self.percentages),"currency_values":", ".join(self.currency_values),"events":", ".join(self.events),"sentiment_label":self.sentiment.label,"sentiment_score":self.sentiment.score}
