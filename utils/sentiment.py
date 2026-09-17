"""Local FinBERT sentiment inference (PyTorch/Transformers, CPU-only).

This is the pipeline's single, authoritative sentiment implementation.
There is no remote API call and no fallback to a different, non-financial
model: if the FinBERT model can't be loaded, inference fails loudly (see
finbert_infer) and the caller (app.services.SentimentService) turns that
into a controlled per-article error state -- it never silently reports a
different model's output as if it were FinBERT.

Design notes:
  * Lazy loading -- torch/transformers are imported, and the model is
    downloaded/loaded, only inside _load_model(), which only runs on the
    first real inference call. Importing this module (e.g. transitively via
    app.services) or running with --no-sentiment never touches torch or
    transformers at all.
  * Model caching -- _load_model() is wrapped in functools.lru_cache(1), so
    the tokenizer/model pair is loaded exactly once per process no matter
    how many articles are scored.
  * Long articles -- FinBERT's 512-token limit is handled by chunking with
    overlap (via the tokenizer's return_overflowing_tokens) rather than
    hard truncation, so a long article's sentiment reflects the whole text,
    not just its first ~512 tokens. Per-chunk probabilities are averaged.
  * CPU only -- the device is fixed to CPU; no CUDA is required or assumed.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache

from config import FINBERT_MAX_TOKENS, FINBERT_MODEL_NAME

LOG = logging.getLogger(__name__)

_CTL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_WS_RE = re.compile(r"\s+")

# The three labels this module knows how to score. This is NOT an assumption
# about which logit index maps to which label -- that mapping is read from
# the loaded model's own `model.config.id2label` at load time (see
# _load_model). This tuple only defines the *complete set* of labels
# finbert_infer can produce; if a model's id2label doesn't contain exactly
# these three (case-insensitively), loading fails loudly rather than
# silently scoring against the wrong labels.
_EXPECTED_LABELS = frozenset({"positive", "negative", "neutral"})

# Token overlap between consecutive chunks of a long article, so a
# sentiment-bearing phrase that straddles a chunk boundary isn't cut in
# half for both halves. Small and fixed -- an implementation detail of the
# chunking strategy, not a tunable knob.
_CHUNK_OVERLAP_TOKENS = 32


@dataclass(frozen=True)
class SentimentResult:
    """Result of a local FinBERT inference call."""
    label: str          # "positive" | "negative" | "neutral"
    score: float         # signed [-1, 1]: positive_prob - negative_prob
    model: str            # exact model identifier used (for logging/traceability)


def clean_for_inference(text: str) -> str:
    """Strip control characters and collapse whitespace before tokenizing."""
    if not text:
        return ""
    text = _CTL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _id2label_from_config(id2label_raw) -> tuple[str, ...]:
    """Turn a model's `config.id2label` into a position-indexed label tuple.

    This is the fix for the inverted-sentiment bug: label order is read
    from the model's own config (whose keys may be int or str, depending on
    how the config was loaded) rather than assumed. Raises ValueError if the
    resulting label set isn't exactly {positive, negative, neutral} --
    scoring would otherwise silently use the wrong keys (see
    _scores_to_result), which is worse than failing loudly here.
    """
    if not id2label_raw:
        raise ValueError("Model config has no id2label mapping")
    by_index = {int(idx): str(label).strip().lower() for idx, label in id2label_raw.items()}
    ordered = tuple(by_index[i] for i in sorted(by_index))
    if set(ordered) != _EXPECTED_LABELS:
        raise ValueError(
            f"Unexpected FinBERT label set {sorted(set(ordered))!r}; "
            f"expected exactly {sorted(_EXPECTED_LABELS)!r}. Refusing to score "
            f"against an unverified label mapping."
        )
    return ordered


class _ModelBundle:
    """A loaded tokenizer/model pair, built once and reused for the process."""
    __slots__ = ("tokenizer", "model", "device", "model_name", "id2label")

    def __init__(self, tokenizer, model, device, model_name, id2label=None):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model_name = model_name
        # Only used as a fallback for bundles built outside _load_model
        # (e.g. hand-constructed in tests); the real load path always
        # supplies the model's actual, verified id2label explicitly.
        self.id2label = id2label or ("positive", "negative", "neutral")


@lru_cache(maxsize=1)
def _load_model() -> _ModelBundle:
    """Load the FinBERT tokenizer/model once per process, on CPU.

    torch/transformers are imported here, not at module scope, so this
    function -- and the cost/side-effects of loading a model -- only runs
    when sentiment inference is actually requested.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    LOG.info("Loading FinBERT model %r (CPU)", FINBERT_MODEL_NAME)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    id2label = _id2label_from_config(dict(model.config.id2label))
    LOG.info(
        "FinBERT model %r loaded in %.2fs on %s -- id2label=%s",
        FINBERT_MODEL_NAME, time.monotonic() - started, device, dict(enumerate(id2label)),
    )
    return _ModelBundle(tokenizer, model, device, FINBERT_MODEL_NAME, id2label=id2label)


def _softmax_row_to_scores(row, labels: tuple[str, ...]) -> dict:
    return {label: float(row[i]) for i, label in enumerate(labels)}


def _aggregate_chunks(chunk_scores: list[dict], labels: tuple[str, ...]) -> dict:
    """Average per-label PROBABILITIES across an article's chunks.

    This averages raw probabilities from every chunk first; the final
    label/score is derived exactly once, afterward, from the aggregate (see
    _scores_to_result / finbert_infer) -- never by averaging already-signed
    per-chunk scores, which would not correspond to any real probability
    distribution and could disagree with the label that's actually most
    probable overall.
    """
    totals = {label: 0.0 for label in labels}
    for scores in chunk_scores:
        for label in labels:
            totals[label] += scores[label]
    n = len(chunk_scores) or 1
    return {label: totals[label] / n for label in labels}


def _scores_to_result(scores: dict, model_name: str) -> SentimentResult:
    label = max(scores, key=scores.get)
    score = scores["positive"] - scores["negative"]
    return SentimentResult(label=label, score=score, model=model_name)


def finbert_infer(text: str) -> SentimentResult:
    """Run local FinBERT inference on one article's text.

    Raises on model-load or inference failure -- callers are responsible
    for catching that and reporting a controlled per-article error state
    (see app.services.SentimentService.analyze) so one bad article can't
    take down the run.
    """
    import torch

    cleaned = clean_for_inference(text)
    bundle = _load_model()
    if not cleaned:
        return SentimentResult(label="neutral", score=0.0, model=bundle.model_name)

    # `return_overflowing_tokens` + `stride` makes the tokenizer itself
    # split long input into overlapping <=512-token windows; short text
    # simply comes back as a single window. Either way we get a uniform
    # (num_chunks, seq_len) batch for one forward pass.
    encoded = bundle.tokenizer(
        cleaned,
        truncation=True,
        max_length=FINBERT_MAX_TOKENS,
        stride=_CHUNK_OVERLAP_TOKENS,
        return_overflowing_tokens=True,
        return_tensors="pt",
        padding=True,
    )
    input_ids = encoded["input_ids"].to(bundle.device)
    attention_mask = encoded["attention_mask"].to(bundle.device)

    with torch.no_grad():
        outputs = bundle.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

    chunk_scores = [_softmax_row_to_scores(row, bundle.id2label) for row in probs]
    aggregated = _aggregate_chunks(chunk_scores, bundle.id2label)
    result = _scores_to_result(aggregated, bundle.model_name)
    LOG.debug(
        "FinBERT (%s) scored %d chunk(s): %s (%.4f)",
        bundle.model_name, len(chunk_scores), result.label, result.score,
    )
    return result
