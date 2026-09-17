"""Real (non-mocked) FinBERT diagnostics.

Run this on the target machine with torch/transformers installed:

    python -m scripts.finbert_diagnostics
    python scripts/finbert_diagnostics.py

Prints, using the real loaded model, no mocks:
  1. The model's actual config.id2label mapping.
  2. Raw probabilities/label/score for three unambiguous sanity texts
     (positive/negative/neutral) -- the obvious positive text must not
     come out negative, and vice versa.
  3. The same, for the specific real article texts reported as
     misclassified, so the actual text sent into FinBERT is visible
     alongside its score.

This is diagnostic tooling, not a test -- it doesn't assert anything, it
prints, so you can see the real numbers directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

# See scripts/finbert_smoke_test.py for why this is needed for direct
# (`python scripts/finbert_diagnostics.py`) invocation to work without an
# installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.sentiment import _load_model, clean_for_inference, finbert_infer

SANITY_TEXTS = {
    "positive": (
        "The company reported record revenue and earnings, beat analyst "
        "expectations, raised guidance and shares surged."
    ),
    "negative": (
        "The company reported a major loss, missed analyst expectations, "
        "cut guidance and shares plunged."
    ),
    "neutral": (
        "The company announced that its quarterly earnings report will be "
        "released next week."
    ),
}

# The specific real articles reported as misclassified. Fill in the exact
# text your pipeline actually extracted for these (title + summary, or the
# full extracted article body) if you want a byte-for-byte match to what
# was scored in your CSV -- these are reconstructed from the headlines in
# the report as a starting point.
REPORTED_ARTICLES = {
    "Salesforce rockets 22%": (
        "Salesforce rockets 22% for second-best day ever, leading software rally. "
        "Salesforce shares jumped after the company posted an earnings beat and "
        "expanded its partnership with Anthropic, fueling a broad software rally."
    ),
    "CrowdStrike posts best day ever": (
        "CrowdStrike posts best day ever, Okta's stock pops nearly 29%. "
        "CrowdStrike reported strong earnings and raised its forecast, sending "
        "shares up sharply alongside gains in Okta."
    ),
    "Nvidia/Salesforce earnings upended bear narratives": (
        "Jim Cramer says Nvidia and Salesforce earnings upended two bear narratives. "
        "Cramer pointed to strong results and rallies in both stocks as evidence "
        "against the bearish case on AI and enterprise software spending."
    ),
    "Gap shares jump 12%": (
        "Gap shares jump 12% after quarterly results beat expectations, though "
        "the company flagged mixed performance across some of its brands."
    ),
}


def _print_result(label_text: str, text: str) -> None:
    print(f"\n--- {label_text} ---")
    print(f"Text sent to FinBERT (first 500 chars): {clean_for_inference(text)[:500]!r}")
    result = finbert_infer(text)
    print(f"Label: {result.label}")
    print(f"Score (positive_prob - negative_prob): {result.score:.4f}")
    print(f"Model: {result.model}")


def main() -> None:
    bundle = _load_model()
    print(f"Model: {bundle.model_name}")
    print(f"id2label (by position, as read from the model's own config): {dict(enumerate(bundle.id2label))}")

    print("\n=== Sanity checks (unambiguous texts) ===")
    for label_text, text in SANITY_TEXTS.items():
        _print_result(f"expected: {label_text}", text)

    print("\n\n=== Reported CSV articles ===")
    for headline, text in REPORTED_ARTICLES.items():
        _print_result(headline, text)


if __name__ == "__main__":
    main()
