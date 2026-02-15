"""Unit tests for summarization workload accuracy checking (ROUGE-based)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from workloads.summarization.loader import (
    accuracy_check,
    _compute_rouge,
    _tokenize,
    load_samples,
)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_removes_stopwords(self):
        tokens = _tokenize("the cat is on the mat")
        assert "the" not in tokens
        assert "cat" in tokens
        assert "mat" in tokens

    def test_removes_short_words(self):
        tokens = _tokenize("I am at it")
        assert len(tokens) == 0

    def test_lowercase(self):
        tokens = _tokenize("Machine Learning Model")
        assert "machine" in tokens
        assert "learning" in tokens


# ---------------------------------------------------------------------------
# _compute_rouge
# ---------------------------------------------------------------------------

class TestComputeRouge:
    def test_identical_text(self):
        scores = _compute_rouge("hello world test", "hello world test")
        assert scores["rouge1_f"] == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        scores = _compute_rouge("apple banana cherry", "dog elephant fish")
        assert scores["rouge1_f"] == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        scores = _compute_rouge(
            "LLMs generate text and answer questions",
            "LLMs are used for text generation and question answering",
        )
        assert 0.0 < scores["rouge1_f"] < 1.0

    def test_empty_strings(self):
        scores = _compute_rouge("", "some reference")
        assert scores["rouge1_f"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# accuracy_check (ROUGE-based)
# ---------------------------------------------------------------------------

class TestSummarizationAccuracyCheck:
    def test_good_summary(self):
        ref = "Large language models generate text, summarize documents, and answer questions effectively."
        pred = "Large language models can generate text, summarize documents, and answer questions."
        assert accuracy_check(pred, ref) is True

    def test_empty_prediction(self):
        assert accuracy_check("", "some reference") is False

    def test_empty_reference(self):
        assert accuracy_check("some prediction", "") is False

    def test_too_short(self):
        assert accuracy_check("Hi.", "a longer reference text with content") is False

    def test_unrelated_text(self):
        ref = "Machine learning systems optimize data processing."
        pred = "The weather today is sunny with a high of 75 degrees Fahrenheit."
        assert accuracy_check(pred, ref) is False

    def test_stores_rouge_scores(self):
        ref = "LLMs are versatile tools used for text generation."
        pred = "Large language models generate text effectively."
        accuracy_check(pred, ref)
        scores = accuracy_check.last_rouge_scores
        assert "rouge1_f" in scores
        assert isinstance(scores["rouge1_f"], float)


# ---------------------------------------------------------------------------
# load_samples (toy)
# ---------------------------------------------------------------------------

class TestLoadSamples:
    def test_load_toy(self):
        cfg = {"name": "summarization", "dataset": {"source": "toy", "n_samples": 5}}
        samples = load_samples(cfg)
        assert len(samples) == 5
        assert all(s.text for s in samples)
        assert all(s.reference for s in samples)

    def test_reference_is_not_same_as_text(self):
        """Regression test: references must be actual summaries, not the input text."""
        cfg = {"name": "summarization", "dataset": {"source": "toy", "n_samples": 10}}
        samples = load_samples(cfg)
        for s in samples:
            assert s.reference != s.text, f"Sample {s.sid}: reference should differ from text"

    def test_references_are_shorter(self):
        cfg = {"name": "summarization", "dataset": {"source": "toy", "n_samples": 10}}
        samples = load_samples(cfg)
        for s in samples:
            assert len(s.reference) < len(s.text), (
                f"Sample {s.sid}: reference should be shorter than text"
            )
