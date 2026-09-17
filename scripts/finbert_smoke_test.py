"""Real local FinBERT smoke test -- one genuine inference call, no mocks.

This is deliberately separate from tests/test_sentiment.py (which is fully
mocked and never touches the real model). Run this script once on a machine
with torch/transformers installed and network access to download
ProsusAI/finbert the first time:

    PYTHONPATH=. python3 scripts/finbert_smoke_test.py

It reports exactly what Phase 3 requires: the model identifier, confirmation
the tokenizer and model both loaded, the device used, the sample text,
predicted label, score, and inference time -- using the real
utils.sentiment.finbert_infer(), not a stand-in.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# This project is a flat, non-installed script layout (no setup.py/
# pyproject.toml) -- config.py, app/, and utils/ are resolved relative to
# the project root, not via an installed package. `python -m
# scripts.finbert_smoke_test` gets the project root on sys.path for free
# (that's how -m resolves the parent package); a direct
# `python scripts/finbert_smoke_test.py` invocation does not, since Python
# only puts the script's own directory on sys.path in that mode. This
# insertion makes both invocation styles resolve identically without
# requiring an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FINBERT_MODEL_NAME
from utils.sentiment import _load_model, finbert_infer

SAMPLE_TEXT = (
    "The company reported quarterly revenue well above analyst expectations, "
    "with strong margin expansion and raised full-year guidance."
)


def main() -> None:
    print(f"Model identifier requested: {FINBERT_MODEL_NAME}")

    load_started = time.monotonic()
    bundle = _load_model()
    load_elapsed = time.monotonic() - load_started
    print(f"Tokenizer loaded: {bundle.tokenizer is not None}")
    print(f"Model loaded: {bundle.model is not None}")
    print(f"Device: {bundle.device}")
    print(f"Model load time: {load_elapsed:.2f}s")

    print(f"\nSample text: {SAMPLE_TEXT!r}")
    infer_started = time.monotonic()
    result = finbert_infer(SAMPLE_TEXT)
    infer_elapsed = time.monotonic() - infer_started

    print(f"Predicted label: {result.label}")
    print(f"Score (positive_prob - negative_prob, [-1, 1]): {result.score:.4f}")
    print(f"Model that produced this result: {result.model}")
    print(f"Inference time: {infer_elapsed:.4f}s")


if __name__ == "__main__":
    main()
