"""Unit tests for reasoning workload accuracy checking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from workloads.reasoning.loader import (
    accuracy_check,
    _extract_answer,
    _normalize,
    load_samples,
)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_strip_prefix_answer_is(self):
        assert _normalize("The answer is 42") == "42"

    def test_strip_prefix_therefore(self):
        assert _normalize("Therefore, yes") == "yes"

    def test_strip_trailing_punct(self):
        assert _normalize("42.") == "42"

    def test_lowercase(self):
        assert _normalize("YES") == "yes"

    def test_passthrough(self):
        assert _normalize("Spike") == "spike"


# ---------------------------------------------------------------------------
# _extract_answer
# ---------------------------------------------------------------------------

class TestExtractAnswer:
    def test_hash_format(self):
        assert _extract_answer("some reasoning\n#### 42") == "42"

    def test_answer_is_pattern(self):
        result = _extract_answer("Thinking...\nThe answer is No.")
        assert result is not None
        assert "no" in result.lower()

    def test_boxed(self):
        assert _extract_answer("\\boxed{243}") == "243"

    def test_bold(self):
        result = _extract_answer("So the answer is:\n**Spike**")
        assert result is not None
        assert "Spike" in result

    def test_last_line_fallback(self):
        result = _extract_answer("Some reasoning\nStep 1\nStep 2\n42")
        assert result == "42"


# ---------------------------------------------------------------------------
# accuracy_check
# ---------------------------------------------------------------------------

class TestReasoningAccuracyCheck:
    def test_exact_match(self):
        assert accuracy_check("The answer is 42", "42") is True

    def test_yes_no_match(self):
        assert accuracy_check("After analysis, the answer is No.", "No") is True

    def test_word_boundary_match(self):
        assert accuracy_check("Therefore, Spike is the shortest.", "Spike") is True

    def test_numeric_match(self):
        assert accuracy_check("The result is 243.", "243") is True

    def test_wrong_answer(self):
        assert accuracy_check("The answer is 99", "42") is False

    def test_empty_prediction(self):
        assert accuracy_check("", "42") is False

    def test_case_insensitive(self):
        assert accuracy_check("the answer is YES", "Yes") is True

    def test_boolq_style_yes(self):
        assert accuracy_check("Based on the passage, yes.", "Yes") is True

    def test_boolq_style_no(self):
        assert accuracy_check("No, this is not correct.", "No") is True


# ---------------------------------------------------------------------------
# load_samples (toy)
# ---------------------------------------------------------------------------

class TestLoadSamples:
    def test_load_toy(self):
        cfg = {"name": "reasoning", "dataset": {"source": "toy", "n_samples": 5}}
        samples = load_samples(cfg)
        assert len(samples) == 5
        assert all(s.puzzle for s in samples)
        assert all(s.reference for s in samples)
        assert all(s.puzzle_type for s in samples)

    def test_load_toy_all(self):
        cfg = {"name": "reasoning", "dataset": {"source": "toy", "n_samples": 10}}
        samples = load_samples(cfg)
        assert len(samples) == 10
