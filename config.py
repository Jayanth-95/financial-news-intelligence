"""Application configuration.

Configuration is intentionally centralized so that runtime behaviour can be
changed through environment variables without modifying application code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_TIMEOUT: Final[int] = 20
DEFAULT_USER_AGENT: Final[str] = (
    "news-sentiment-pipeline/1.0 (+research; contact: you@example.com)"
)


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overwriting the environment.

    The project deliberately avoids adding a runtime dependency just for
    loading a local .env file. Existing environment variables always win.
    """
    if not path.is_file():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        # Environment variables remain the source of truth if .env cannot be read.
        return


_load_dotenv(PROJECT_ROOT / ".env")

# The pipeline's single authoritative sentiment model: a genuine,
# finance-specific FinBERT checkpoint, loaded locally via transformers/
# torch (CPU-only). There is no fallback list and no generic
# (non-financial) substitute model -- if this model can't be loaded,
# sentiment inference fails rather than silently answering with a
# different model's output. See utils/sentiment.py.
FINBERT_MODEL_NAME: Final[str] = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")

# ProsusAI/finbert (BERT-base) supports sequences up to 512 tokens
# (including special tokens). Text longer than this is chunked with
# overlap and the per-chunk results are aggregated -- see
# utils/sentiment.py's finbert_infer() for the chunking strategy.
FINBERT_MAX_TOKENS: Final[int] = 512

USER_AGENT: Final[str] = os.getenv("USER_AGENT", DEFAULT_USER_AGENT)
TIMEOUT: Final[int] = max(
    1,
    int(os.getenv("HTTP_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT))),
)
