
# News Sentiment (local FinBERT)

This is a minimal, production-minded starter to collect finance news from **Moneycontrol**, **Financial Times (RSS summaries only)**, **The Block**, and a safe **Google News (Business)** RSS proxy, clean it, filter for stock-market relevance, and score sentiment with **FinBERT** (`ProsusAI/finbert`) running locally via `transformers`/`torch` on CPU. The model is downloaded once from the Hugging Face Hub on first use and cached locally by `transformers`; no API token and no per-request network call are needed for sentiment itself.

> ⚠️ Respect each site's Terms of Service & robots.txt. FT and some publishers are paywalled—this project **only uses their RSS summaries by default**. Scraping paywalled content is out of scope.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python pipeline.py --days 1 --out data/sentiment_latest.csv
```

The first run with sentiment enabled will download the `ProsusAI/finbert` weights (a few hundred MB) from the Hugging Face Hub; subsequent runs load the cached copy. Use `--no-sentiment` if you don't want that download at all -- see below.

- Output CSV columns: `source, url, published, title, summary, text, tickers, sentiment_label, sentiment_score`

Note: newer runs also include extracted financial entities to help downstream analysis.
New CSV columns added: `companies, tickers, percentages, currency_values, events` in addition to the previous fields (`sentiment_label`, `sentiment_score`).

## The `--days N` window

Each collected item's publication date is checked against a rolling `N`-day
window (see `utils/dates.py`) before it's accepted as a *new* row -- an
article published outside that window is skipped during collection. This
only gates new articles; it never deletes or filters rows already present
in the output CSV (see "No duplicates" below). Supported date formats:
RFC 2822 (typical RSS `<pubDate>`), ISO 8601/RFC 3339 (including a
trailing `Z`), and naive timestamps (assumed UTC). An article with a
missing or unparseable publication date is accepted rather than rejected
-- some source adapters don't always have a reliable date to report, and
silently dropping everything without one would lose legitimate current
content. An article with a *known* date outside the window is always
rejected regardless of this default.

## Design
- **Ingestion:** RSS feeds for Moneycontrol, FT (summaries only), The Block, Google News (Business) as a proxy for Google Finance.
- **Extraction:** `trafilatura` to fetch & extract main text when allowed; otherwise fallback to title/summary.
- **Filtering:** Keep items that look stock-market relevant via keyword heuristics (earnings, guidance, shares, stock, IPO, etc.).
- **Sentiment:** FinBERT (`ProsusAI/finbert`) loaded locally via `transformers`/`torch`, CPU-only. The model/tokenizer are loaded lazily on first use and cached for the life of the process (see `utils/sentiment.py`); there is a single authoritative implementation and no fallback to a different, non-financial model. Long articles are chunked with overlap and the per-chunk results are averaged, rather than being truncated to the first ~512 tokens.
- **No duplicates:** Simple hash on (source, title); existing output rows are preserved and merged across runs (see `app/runner.py`). The `--days` window (above) controls which *new* articles are accepted during collection -- it never deletes previously persisted rows.
- **Extensible:** Add more sources under `sources/`.

## Notes on the requested sources
- **Moneycontrol:** Uses public RSS feeds; safe to ingest.
- **Financial Times:** Paywalled; we only use RSS metadata/summary by default.
- **The Block:** Public RSS available.
- **Google Finance:** No official public RSS and scraping can violate TOS. We use **Google News - Business RSS** as a safe proxy for broad market news.

You can later swap in paid/licensed feeds (e.g., GDELT, NewsAPI, Refinitiv) for reliability.

## New options & offline test

The pipeline now supports several command-line options:

- `--no-sentiment`: skip FinBERT calls and store `sentiment_label=skipped`.
- `--mock-sentiment`: set sentiment to neutral (useful for offline testing).
- `--ticker-map PATH`: CSV file mapping Company,Ticker used for fuzzy ticker mapping (default: `data/ticker_mapping.csv`).
- `--no-spacy`: disable spaCy NER (the code will fall back to heuristics).

For a quick offline dry-run that doesn't load FinBERT or spaCy, run the included sample harness:

```bash
python sample_run.py
```

This writes `data/sample_output.csv` using two example articles and mocked sentiment.
