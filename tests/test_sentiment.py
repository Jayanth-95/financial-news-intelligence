"""Mocked unit tests for local FinBERT sentiment inference (audit D7, G).

None of these tests download or run the real ProsusAI/finbert model --
tokenizer/model objects are faked, and utils.sentiment._load_model is
monkeypatched in the inference tests so finbert_infer() runs against
controlled, deterministic fakes. See tests/test_sentiment_smoke.py (a
separate, explicitly-real-model script, not part of this suite) for actual
local model inference.

Run with: PYTHONPATH=. python3 tests/test_sentiment.py
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import utils.sentiment as sentiment_mod
from app.models import ArticleAnalysis, NewsItem, Sentiment
from app.services import SentimentService
from utils.sentiment import SentimentResult, finbert_infer

PROJECT_ROOT = Path(sentiment_mod.__file__).resolve().parents[1]


# --- Fakes: stand in for the HF tokenizer/model contract, not for torch ---

class _FakeTokenizer:
    """Mimics the subset of the HF tokenizer __call__ contract finbert_infer
    relies on: accepts text plus the chunking kwargs, returns a dict with
    input_ids/attention_mask shaped (n_chunks, seq_len).
    """

    def __init__(self, n_chunks=1, seq_len=6):
        self.n_chunks = n_chunks
        self.seq_len = seq_len
        self.last_kwargs = None

    def __call__(self, text, **kwargs):
        self.last_kwargs = kwargs
        ids = torch.tensor([[1] * self.seq_len for _ in range(self.n_chunks)])
        mask = torch.tensor([[1] * self.seq_len for _ in range(self.n_chunks)])
        return {"input_ids": ids, "attention_mask": mask}


class _FakeOutputs:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    """Stands in for the loaded FinBERT model: returns pre-set logits per
    chunk so a test can assert an exact resulting label/score, and counts
    calls so tests can assert batching behavior.
    """

    def __init__(self, logits_per_chunk):
        self.logits_per_chunk = logits_per_chunk
        self.call_count = 0

    def __call__(self, input_ids=None, attention_mask=None):
        self.call_count += 1
        return _FakeOutputs(torch.tensor(self.logits_per_chunk))


def _fake_bundle(logits_per_chunk, n_chunks=None, model_name="fake/finbert-test", id2label=("positive", "negative", "neutral")):
    n_chunks = n_chunks if n_chunks is not None else len(logits_per_chunk)
    return sentiment_mod._ModelBundle(
        _FakeTokenizer(n_chunks=n_chunks), _FakeModel(logits_per_chunk), "cpu", model_name, id2label=id2label
    )


# Logit positions match ProsusAI/finbert's REAL id2label order, read from
# config rather than assumed -- see _id2label_from_config. index0=positive,
# index1=negative, index2=neutral.
_POSITIVE_LOGITS = [[4.0, 0.1, 0.2]]
_NEGATIVE_LOGITS = [[0.1, 4.0, 0.2]]
_NEUTRAL_LOGITS = [[0.1, 0.2, 4.0]]


# --- 1. Service initialization ---------------------------------------------

def test_sentiment_service_initialization():
    svc = SentimentService(enabled=True, mock=False)
    assert svc.enabled is True and svc.mock is False
    svc2 = SentimentService(enabled=False, mock=True)
    assert svc2.enabled is False and svc2.mock is True


# --- 2. Lazy loading ---------------------------------------------------------

def test_no_module_level_torch_or_transformers_import():
    """Structural guarantee: torch/transformers are never imported at
    utils/sentiment.py module scope, only inside functions. This is what
    makes `--no-sentiment` avoid loading the model at all.
    """
    source = Path(sentiment_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:  # top-level statements only
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
            assert "torch" not in names, "torch must not be imported at module scope"
            assert "transformers" not in names, "transformers must not be imported at module scope"
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("torch", "transformers"), node.module


def test_importing_sentiment_module_does_not_load_torch_or_transformers():
    """Clean-interpreter proof: merely importing utils.sentiment never
    pulls in torch/transformers, regardless of whether they're installed.
    """
    script = (
        "import sys; "
        "import utils.sentiment; "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False False", proc.stdout


def test_no_sentiment_never_touches_model():
    def _boom():
        raise AssertionError("_load_model must not be called when sentiment is disabled")
    with patch.object(sentiment_mod, "_load_model", side_effect=_boom):
        result = SentimentService(enabled=False).analyze("Apple reported strong earnings.")
    assert result == Sentiment("skipped", 0.0)


def test_mock_sentiment_never_touches_model():
    def _boom():
        raise AssertionError("_load_model must not be called in mock mode")
    with patch.object(sentiment_mod, "_load_model", side_effect=_boom):
        result = SentimentService(enabled=True, mock=True).analyze("Apple reported strong earnings.")
    assert result == Sentiment("neutral", 0.0)


# --- 3. Cached model reuse ---------------------------------------------------

def test_model_loaded_once_and_cached():
    sentiment_mod._load_model.cache_clear()
    fake_tokenizer = object()
    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    try:
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer) as tok, \
             patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model) as mdl:
            bundle1 = sentiment_mod._load_model()
            bundle2 = sentiment_mod._load_model()
        assert tok.call_count == 1, "tokenizer must load exactly once per process"
        assert mdl.call_count == 1, "model must load exactly once per process"
        assert bundle1 is bundle2, "second call must reuse the cached bundle"
        assert bundle1.model_name == sentiment_mod.FINBERT_MODEL_NAME
        assert bundle1.id2label == ("positive", "negative", "neutral")
    finally:
        sentiment_mod._load_model.cache_clear()


def test_load_model_reads_id2label_from_config_not_assumed():
    """The core Phase 5 regression: label order must come from the model's
    own config, including when that order is NOT the commonly-assumed
    (negative, neutral, positive) sequence.
    """
    sentiment_mod._load_model.cache_clear()
    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    # Deliberately an unusual order, to prove it's read rather than assumed.
    fake_model.config.id2label = {0: "neutral", 1: "positive", 2: "negative"}
    try:
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()), \
             patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model):
            bundle = sentiment_mod._load_model()
        assert bundle.id2label == ("neutral", "positive", "negative")
    finally:
        sentiment_mod._load_model.cache_clear()


def test_load_model_rejects_unexpected_label_set():
    """If a model's id2label doesn't contain exactly positive/negative/
    neutral, loading must fail loudly rather than silently score against
    the wrong keys.
    """
    sentiment_mod._load_model.cache_clear()
    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.config.id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    try:
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()), \
             patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model):
            raised = False
            try:
                sentiment_mod._load_model()
            except ValueError:
                raised = True
            assert raised, "unexpected label set must raise, not silently proceed"
    finally:
        sentiment_mod._load_model.cache_clear()


# --- 4-6. Positive / negative / neutral financial text ----------------------

def test_positive_financial_text():
    bundle = _fake_bundle(_POSITIVE_LOGITS)
    with patch.object(sentiment_mod, "_load_model", return_value=bundle):
        result = finbert_infer("Company beats earnings expectations by a wide margin.")
    assert result.label == "positive", result
    assert result.score > 0
    assert result.model == "fake/finbert-test"


def test_negative_financial_text():
    bundle = _fake_bundle(_NEGATIVE_LOGITS)
    with patch.object(sentiment_mod, "_load_model", return_value=bundle):
        result = finbert_infer("Company misses estimates and issues a profit warning.")
    assert result.label == "negative", result
    assert result.score < 0


def test_neutral_financial_text():
    bundle = _fake_bundle(_NEUTRAL_LOGITS)
    with patch.object(sentiment_mod, "_load_model", return_value=bundle):
        result = finbert_infer("Company will report quarterly results next week.")
    assert result.label == "neutral", result


def test_empty_text_short_circuits_without_model_call():
    bundle = _fake_bundle(_POSITIVE_LOGITS)
    with patch.object(sentiment_mod, "_load_model", return_value=bundle):
        result = finbert_infer("   ")
    assert result == SentimentResult("neutral", 0.0, "fake/finbert-test")
    assert bundle.model.call_count == 0, "model must never be invoked for empty text"


# --- 7. Long article handling (chunking + aggregation) ----------------------

def test_long_article_is_chunked_and_averaged_in_one_batch():
    # chunk 1 strongly positive (index0), chunk 2 strongly negative (index1)
    # -> averaged probabilities should roughly cancel out.
    logits = [[4.0, 0.1, 0.1], [0.1, 4.0, 0.1]]
    bundle = _fake_bundle(logits, n_chunks=2)
    long_text = "quarterly results outlook guidance " * 400
    with patch.object(sentiment_mod, "_load_model", return_value=bundle):
        result = finbert_infer(long_text)
    assert bundle.model.call_count == 1, "chunks must be scored in a single batched forward pass"
    assert abs(result.score) < 0.01, f"averaged score should roughly cancel out, got {result.score}"


def test_chunk_aggregation_averages_probabilities_not_signed_scores():
    """Requirement E: the aggregate label/score must come from averaging
    per-chunk PROBABILITIES once, not from averaging already-computed
    per-chunk signed scores. Three chunks, two dominant-positive and one
    dominant-negative: averaging probabilities correctly yields an overall
    positive result. (Averaging signed per-chunk scores computed
    independently would happen to give the same answer here by symmetry --
    the real point is that finbert_infer only ever calls _scores_to_result
    ONCE, on the aggregate, which test_long_article_... already confirms via
    a single model call; this test checks the arithmetic directly.)
    """
    chunk_scores = [
        {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
        {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
        {"positive": 0.05, "negative": 0.9, "neutral": 0.05},
    ]
    aggregated = sentiment_mod._aggregate_chunks(chunk_scores, ("positive", "negative", "neutral"))
    expected_positive = (0.9 + 0.9 + 0.05) / 3
    expected_negative = (0.05 + 0.05 + 0.9) / 3
    assert abs(aggregated["positive"] - expected_positive) < 1e-9
    assert abs(aggregated["negative"] - expected_negative) < 1e-9
    result = sentiment_mod._scores_to_result(aggregated, "fake/finbert-test")
    assert result.label == "positive"
    assert abs(result.score - (expected_positive - expected_negative)) < 1e-9


def test_short_article_is_a_single_chunk():
    bundle = _fake_bundle(_POSITIVE_LOGITS, n_chunks=1)
    with patch.object(sentiment_mod, "_load_model", return_value=bundle):
        finbert_infer("Short headline about strong earnings.")
    assert bundle.tokenizer.last_kwargs["return_overflowing_tokens"] is True
    assert bundle.tokenizer.last_kwargs["max_length"] == sentiment_mod.FINBERT_MAX_TOKENS


# --- 8. Inference failure isolation ------------------------------------------

def test_inference_failure_returns_controlled_error_state():
    class _BoomModel:
        def __call__(self, **kwargs):
            raise RuntimeError("simulated inference crash")

    bundle = sentiment_mod._ModelBundle(_FakeTokenizer(), _BoomModel(), "cpu", "fake/finbert-test")
    with patch.object(sentiment_mod, "_load_model", return_value=bundle):
        result = SentimentService(enabled=True, mock=False).analyze("Some article text.")
    assert result == Sentiment("error", 0.0)


def test_model_load_failure_also_isolated():
    with patch.object(sentiment_mod, "_load_model", side_effect=RuntimeError("no weights found")):
        result = SentimentService(enabled=True, mock=False).analyze("Some article text.")
    assert result == Sentiment("error", 0.0)


# --- 9. No generic fallback model, no duplicate implementation --------------

def test_no_generic_fallback_model_referenced_anywhere():
    for relative in ("utils/sentiment.py", "config.py", "app/services.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8").lower()
        assert "cardiffnlp" not in source, relative
        assert "twitter-roberta" not in source, relative


def test_disconnected_duplicate_implementation_removed():
    assert not (PROJECT_ROOT / "utils" / "finbert_local.py").exists()
    assert not (PROJECT_ROOT / "utils" / "analyze_sentiment_csv.py").exists()


def test_authoritative_model_is_genuine_finbert():
    assert "finbert" in sentiment_mod.FINBERT_MODEL_NAME.lower()


# --- 10. --no-sentiment CSV/output contract unchanged ------------------------

def test_csv_schema_unaffected_by_sentiment_change():
    item = NewsItem(source="s", title="t")
    analysis = ArticleAnalysis(item=item, text="t", sentiment=Sentiment("positive", 0.42))
    row = analysis.as_row()
    assert set(row.keys()) == {
        "source", "url", "published", "title", "summary", "text",
        "companies", "tickers", "mapped_tickers", "percentages",
        "currency_values", "events", "sentiment_label", "sentiment_score",
    }
    assert row["sentiment_label"] == "positive"
    assert row["sentiment_score"] == 0.42


def test_no_sentiment_flag_still_produces_skipped():
    row = ArticleAnalysis(item=NewsItem(source="s", title="t"), text="t").as_row()
    assert row["sentiment_label"] == "skipped"
    assert row["sentiment_score"] == 0.0


TESTS = [
    test_sentiment_service_initialization,
    test_no_module_level_torch_or_transformers_import,
    test_importing_sentiment_module_does_not_load_torch_or_transformers,
    test_no_sentiment_never_touches_model,
    test_mock_sentiment_never_touches_model,
    test_model_loaded_once_and_cached,
    test_load_model_reads_id2label_from_config_not_assumed,
    test_load_model_rejects_unexpected_label_set,
    test_positive_financial_text,
    test_negative_financial_text,
    test_neutral_financial_text,
    test_empty_text_short_circuits_without_model_call,
    test_long_article_is_chunked_and_averaged_in_one_batch,
    test_chunk_aggregation_averages_probabilities_not_signed_scores,
    test_short_article_is_a_single_chunk,
    test_inference_failure_returns_controlled_error_state,
    test_model_load_failure_also_isolated,
    test_no_generic_fallback_model_referenced_anywhere,
    test_disconnected_duplicate_implementation_removed,
    test_authoritative_model_is_genuine_finbert,
    test_csv_schema_unaffected_by_sentiment_change,
    test_no_sentiment_flag_still_produces_skipped,
]


def run():
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    run()
