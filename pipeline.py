"""CLI entry point; application implementation lives under app/."""
import argparse, logging, csv
from pathlib import Path
from app.models import NewsItem
from app.runner import PipelineRunner
from app.services import AnalysisOptions,ArticleAnalyzer,ArticleExtractor,SentimentService,TickerResolver,load_ticker_mapping,article_key
from sources.moneycontrol import fetch_items as fetch_mc
from sources.ft import fetch_items as fetch_ft
from sources.theblock import fetch_items as fetch_tb
from sources.google_news_business import fetch_items as fetch_gnb
from sources.cnbc import fetch_items as fetch_cnbc
from sources.nasdaq import fetch_items as fetch_nasdaq
from sources.reuters_feeds import fetch_items as fetch_reuters
from sources.marketwatch import fetch_items as fetch_marketwatch
from sources.investing import fetch_items as fetch_investing
from sources.reddit import fetch_items as fetch_reddit
SOURCES=(fetch_mc,fetch_ft,fetch_tb,fetch_gnb,fetch_cnbc,fetch_nasdaq,fetch_reuters,fetch_marketwatch,fetch_investing,fetch_reddit)

# Resolved relative to THIS FILE, not the current working directory.
# `python pipeline.py`, `python /some/other/path/pipeline.py`, and
# `python -m pipeline` from any directory must all find
# data/ticker_mapping.csv and default to writing data/sentiment_latest.csv
# in the project itself -- a bare relative string default
# ("data/ticker_mapping.csv") silently resolves against whatever directory
# the command happens to be invoked from, which previously caused the
# ticker map to load as empty (with only a debug-level, easy-to-miss log
# line) whenever the pipeline was run from anywhere else. An explicit
# --ticker-map/--out argument from the user is still honored as given,
# resolved the normal way (relative to the invocation's cwd) -- only the
# *default* needs to be location-independent.
PROJECT_ROOT=Path(__file__).resolve().parent
DEFAULT_TICKER_MAP=str(PROJECT_ROOT/"data"/"ticker_mapping.csv")
DEFAULT_OUTPUT=str(PROJECT_ROOT/"data"/"sentiment_latest.csv")

def hash_key(source,title): return article_key(NewsItem(source,title))
def load_ticker_map(path): return load_ticker_mapping(path)
def map_tickers(companies,ticker_map): return TickerResolver(ticker_map).resolve(companies)
def load_seen_keys(path):
    p=Path(path)
    if not p.exists(): return set()
    with p.open(newline="",encoding="utf-8") as f: return {hash_key(r.get("source",""),r.get("title","")) for r in csv.DictReader(f) if r.get("source") or r.get("title")}
def parse_args(argv=None):
    p=argparse.ArgumentParser(description="Collect and analyze financial news."); p.add_argument("--days",type=int,default=1); p.add_argument("--out",default=DEFAULT_OUTPUT); p.add_argument("--no-sentiment",action="store_true"); p.add_argument("--mock-sentiment",action="store_true"); p.add_argument("--ticker-map",default=DEFAULT_TICKER_MAP); p.add_argument("--no-spacy",action="store_true"); return p.parse_args(argv)
def run(days=1,out=None,use_sentiment=True,mock_sentiment=False,ticker_map=None,use_spacy=True):
    out=out if out is not None else DEFAULT_OUTPUT; ticker_map=ticker_map if ticker_map is not None else DEFAULT_TICKER_MAP
    analyzer=ArticleAnalyzer(ArticleExtractor(),TickerResolver(load_ticker_mapping(ticker_map)),SentimentService(use_sentiment,mock_sentiment),AnalysisOptions(use_spacy)); return PipelineRunner(SOURCES,analyzer,out,days).execute()
def main(argv=None):
    a=parse_args(argv); logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"); return run(a.days,a.out,not a.no_sentiment,a.mock_sentiment,a.ticker_map,not a.no_spacy)
if __name__=="__main__": main()
