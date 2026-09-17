# \# Financial News Intelligence \& Sentiment Analysis

# 

# An end-to-end financial news intelligence pipeline that collects financial news from multiple configured sources, extracts article content, filters for financial relevance, identifies companies and stock tickers, resolves ticker mappings, extracts financial events and quantitative information, and analyzes article sentiment using the local `ProsusAI/finbert` model.

# 

# The pipeline uses Python, `transformers`, `torch`, spaCy, Trafilatura, rule-based financial entity extraction, and configurable ticker mappings. FinBERT runs locally on CPU, with lazy model loading, long-article chunking, and failure isolation.

# 

# The resulting dataset is stored as structured CSV data containing article metadata, companies, tickers, mapped tickers, percentages, monetary values, financial events, sentiment labels, and sentiment scores.

# 

# \## Quick Start

# 

# \### Windows

# 

# ```bat

# python -m venv .venv

# .venv\\Scripts\\activate

# pip install -r requirements.txt

# python pipeline.py --days 1 --out data/sentiment\_latest.csv

# ```

# 

# \### Linux / macOS

# 

# ```bash

# python -m venv .venv

# source .venv/bin/activate

# pip install -r requirements.txt

# python pipeline.py --days 1 --out data/sentiment\_latest.csv

# ```

# 

# On the first sentiment-enabled run, the `ProsusAI/finbert` model weights are downloaded from the Hugging Face Hub and cached locally by `transformers`. Subsequent runs can reuse the cached model.

# 

# Use `--no-sentiment` when you want to skip FinBERT processing.

# 

# \## Pipeline Architecture

# 

# ```text

# News Sources

# &#x20;    |

# &#x20;    v

# Article Collection

# &#x20;    |

# &#x20;    v

# Web Content Extraction

# &#x20;    |

# &#x20;    v

# Financial Relevance Filtering

# &#x20;    |

# &#x20;    v

# Company / Entity Extraction

# &#x20;    |

# &#x20;    v

# Ticker Extraction \& Resolution

# &#x20;    |

# &#x20;    v

# Financial Event Extraction

# &#x20;    |

# &#x20;    +--> Percentages

# &#x20;    |

# &#x20;    +--> Currency Values

# &#x20;    |

# &#x20;    v

# FinBERT Sentiment Analysis

# &#x20;    |

# &#x20;    v

# Persistence + Deduplication

# &#x20;    |

# &#x20;    v

# Structured CSV Output

# ```

# 

# \## Main Features

# 

# \### Multi-source Financial News Ingestion

# 

# The project contains source adapters for configured financial-news feeds and web sources, including:

# 

# \- CNBC

# \- Financial Times RSS

# \- The Block

# \- Investing.com

# \- Nasdaq

# \- MarketWatch

# \- Moneycontrol

# \- Reuters feeds

# \- Google News Business

# \- Reddit

# 

# Source availability depends on feed status, publisher restrictions, network conditions, and the current configuration.

# 

# \### Article Extraction

# 

# `trafilatura` is used to retrieve and extract main article content where access is available.

# 

# When full-page extraction is unavailable, the pipeline can fall back to available title and summary information rather than stopping the entire run.

# 

# \### Financial Relevance Filtering

# 

# The pipeline applies financial relevance heuristics using terms such as:

# 

# ```text

# earnings

# guidance

# IPO

# dividend

# buyback

# shares

# stock

# market

# profit

# loss

# revenue

# forecast

# merger

# acquisition

# hedge

# fund

# portfolio

# investment

# investor

# trading

# stake

# holding

# capital

# assets

# private equity

# venture capital

# ```

# 

# This helps reduce unrelated content before deeper analysis.

# 

# \### Company Extraction

# 

# Company recognition uses a hybrid approach:

# 

# \- legal company-name patterns

# \- possessive-name patterns

# \- curated known-company names

# \- ticker-aware validation

# \- spaCy Named Entity Recognition

# \- structural filtering

# \- conflict handling

# 

# The system deliberately avoids treating every capitalized phrase as a company because financial articles contain people, locations, institutions, products, dates, and other non-company entities.

# 

# \### Primary-Zone Analysis

# 

# The article title, summary, and the beginning of the article receive additional attention because they often contain the most relevant company references.

# 

# The primary zone is processed separately and validated company results from the primary zone are merged with broader article-level extraction.

# 

# \### Ticker Extraction and Resolution

# 

# Ticker symbols are extracted independently from company names and can then be resolved through several levels:

# 

# ```text

# Exact company match

# &#x20;       |

# &#x20;       v

# Alias match

# &#x20;       |

# &#x20;       v

# Known ticker match

# &#x20;       |

# &#x20;       v

# Conservative fuzzy matching

# ```

# 

# Ticker mappings are stored in:

# 

# ```text

# data/ticker\_mapping.csv

# ```

# 

# This allows company-to-ticker mappings to be updated without modifying the Python source code.

# 

# \### Financial Event Extraction

# 

# The pipeline detects financial events such as:

# 

# ```text

# IPO

# dividend

# acquisition

# merger

# ```

# 

# \### Quantitative Extraction

# 

# The parser extracts:

# 

# \- percentages

# \- currency values

# 

# Examples:

# 

# ```text

# 10%

# 12.5%

# $5 billion

# $61 million

# £100 million

# ```

# 

# \## Sentiment Analysis

# 

# The project uses:

# 

# ```text

# ProsusAI/finbert

# ```

# 

# for financial-domain sentiment analysis.

# 

# FinBERT classifies financial text as:

# 

# ```text

# positive

# negative

# neutral

# ```

# 

# The pipeline also stores a numerical sentiment score.

# 

# \### Local Inference

# 

# FinBERT runs locally through:

# 

# ```text

# PyTorch

# Hugging Face Transformers

# CPU

# ```

# 

# The model is loaded lazily and cached during the process.

# 

# There is no per-article sentiment API call.

# 

# \### Long Article Handling

# 

# Long financial articles can exceed transformer input limits. Instead of simply truncating them, the pipeline uses overlapping chunks:

# 

# ```text

# Long Article

# &#x20;   |

# &#x20;   v

# Tokenizer

# &#x20;   |

# &#x20;   v

# Overlapping Chunks

# &#x20;   |

# &#x20;   v

# FinBERT Inference

# &#x20;   |

# &#x20;   v

# Average Chunk Probabilities

# &#x20;   |

# &#x20;   v

# Final Sentiment

# ```

# 

# This allows more of the article to contribute to the final sentiment result.

# 

# \### Failure Isolation

# 

# Errors affecting one source or article are logged and isolated where possible so that other articles can continue to be processed.

# 

# \## Output

# 

# The pipeline writes its primary results to:

# 

# ```text

# data/sentiment\_latest.csv

# ```

# 

# Current output columns:

# 

# ```text

# source

# url

# published

# title

# summary

# text

# companies

# tickers

# mapped\_tickers

# percentages

# currency\_values

# events

# sentiment\_label

# sentiment\_score

# ```

# 

# \### Example

# 

# A processed article can produce information such as:

# 

# ```text

# companies: Anthropic

# tickers: NVDA

# mapped\_tickers: NVDA

# percentages: 37%, 75%

# currency\_values: $965 billion, $2 trillion

# events: ipo

# sentiment\_label: negative

# sentiment\_score: -0.45

# ```

# 

# \## Persistence and Deduplication

# 

# The runner preserves existing output rows across executions.

# 

# The process is approximately:

# 

# ```text

# Load existing CSV

# &#x20;      |

# &#x20;      v

# Collect new articles

# &#x20;      |

# &#x20;      v

# Identify duplicate article keys

# &#x20;      |

# &#x20;      v

# Merge existing + new rows

# &#x20;      |

# &#x20;      v

# Write updated dataset

# ```

# 

# The `--days N` option controls the date window for newly accepted articles.

# 

# It does not delete previously persisted rows.

# 

# The system uses stable article keys to prevent the same article from being repeatedly inserted on subsequent runs.

# 

# Output writing uses a temporary file before replacing the target output, reducing the risk of leaving a partially written CSV after an interrupted write.

# 

# \## The `--days N` Window

# 

# Each collected item's publication date is checked against the configured rolling `N`-day window before being accepted as a new row.

# 

# Supported publication-date formats include common RSS and ISO-style timestamps.

# 

# The window affects newly collected articles only. Existing persisted rows are preserved.

# 

# \## Command-Line Options

# 

# \### `--days N`

# 

# Process newly collected articles inside an `N`-day rolling window.

# 

# ```bat

# python pipeline.py --days 1

# ```

# 

# \### `--no-sentiment`

# 

# Skip FinBERT processing.

# 

# ```bat

# python pipeline.py --days 1 --no-sentiment

# ```

# 

# \### `--mock-sentiment`

# 

# Use mocked neutral sentiment for testing and offline execution.

# 

# ```bat

# python pipeline.py --days 1 --mock-sentiment

# ```

# 

# \### `--ticker-map PATH`

# 

# Use a custom company-to-ticker mapping file.

# 

# ```bat

# python pipeline.py --ticker-map data/ticker\_mapping.csv

# ```

# 

# \### `--no-spacy`

# 

# Disable spaCy NER and use the remaining extraction heuristics.

# 

# ```bat

# python pipeline.py --days 1 --no-spacy

# ```

# 

# \## Offline Test Harness

# 

# The project includes:

# 

# ```text

# sample\_run.py

# ```

# 

# Run:

# 

# ```bat

# python sample\_run.py

# ```

# 

# This performs a small offline-style demonstration using example articles and mocked sentiment.

# 

# It produces:

# 

# ```text

# data/sample\_output.csv

# ```

# 

# \## Testing

# 

# The project includes automated tests covering:

# 

# \- company and entity extraction

# \- ticker extraction and resolution

# \- financial relevance filtering

# \- sentiment processing

# \- persistence

# \- duplicate handling

# \- path and configuration behavior

# 

# Current test status:

# 

# ```text

# 143 passed

# ```

# 

# \## Project Structure

# 

# ```text

# fin/

# |

# +-- app/

# |   +-- models.py

# |   +-- runner.py

# |   +-- services.py

# |

# +-- data/

# |   +-- ticker\_mapping.csv

# |   +-- sentiment\_latest.csv

# |

# +-- scripts/

# |   +-- diagnose\_extraction.py

# |   +-- finbert\_diagnostics.py

# |   +-- finbert\_smoke\_test.py

# |

# +-- sources/

# |   +-- cnbc.py

# |   +-- ft.py

# |   +-- google\_news\_business.py

# |   +-- investing.py

# |   +-- marketwatch.py

# |   +-- moneycontrol.py

# |   +-- nasdaq.py

# |   +-- reddit.py

# |   +-- reuters\_feeds.py

# |   +-- theblock.py

# |

# +-- tests/

# |

# +-- utils/

# |   +-- dates.py

# |   +-- entities.py

# |   +-- extract.py

# |   +-- filtering.py

# |   +-- ner.py

# |   +-- network.py

# |   +-- parse.py

# |   +-- sentiment.py

# |

# +-- config.py

# +-- pipeline.py

# +-- requirements.txt

# +-- sample\_run.py

# +-- README.md

# ```

# 

# \## Data and Publisher Access

# 

# Publisher access and permitted usage vary by source.

# 

# Some sites may be paywalled, rate-limited, protected by anti-bot systems, dynamically rendered, or otherwise restrict automated access.

# 

# The project should be used in accordance with applicable publisher Terms of Service, robots.txt requirements, licenses, and feed/API terms.

# 

# Financial Times content is handled through configured RSS metadata and summaries by default rather than attempting to bypass paywalls.

# 

# For production deployments, licensed or official APIs and feeds can be substituted where appropriate.

# 

# \## Limitations

# 

# Real-world financial news is highly variable.

# 

# Known limitations include:

# 

# \- some publishers restrict automated requests

# \- article extraction can fail on protected or dynamically rendered pages

# \- NER systems can generate false positives

# \- company names may be ambiguous or abbreviated

# \- publisher boilerplate can appear in extracted content

# \- entity extraction depends on available article text and configured company/ticker knowledge

# 

# The pipeline addresses these issues with filtering, validation, fallbacks, and failure isolation, but perfect extraction cannot be guaranteed for every article.

# 

# \## Future Scope

# 

# Possible extensions include:

# 

# \- larger company and ticker knowledge bases

# \- source-specific article parsers

# \- licensed financial-news APIs

# \- PostgreSQL or other database storage

# \- real-time streaming ingestion

# \- historical sentiment analytics

# \- sector-level sentiment aggregation

# \- a web-based financial intelligence dashboard

# \- stronger context-aware entity disambiguation

# \- alerting for important financial events

# 

# \## Technologies

# 

# ```text

# Python 3.13

# PyTorch

# Hugging Face Transformers

# ProsusAI/FinBERT

# spaCy

# Trafilatura

# BeautifulSoup

# Requests

# pytest

# CSV

# RSS

# ```

# 

# \## Responsible Use

# 

# This project is intended for financial-news analysis and research.

# 

# Extracted entities and sentiment scores are automated analytical outputs and should not be treated as guaranteed financial advice or as a substitute for independent research.

