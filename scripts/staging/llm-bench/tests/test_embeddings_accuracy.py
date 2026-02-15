"""Tests for the embeddings (semantic similarity) workload."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workloads.embeddings.loader import (
    _extract_score,
    accuracy_check,
    load_samples,
)


class TestExtractScore:

    def test_plain_number(self):
        assert _extract_score("3.5") == 3.5

    def test_integer(self):
        assert _extract_score("4") == 4.0

    def test_with_text(self):
        assert _extract_score("The similarity score is 2.8.") == 2.8

    def test_clamp_high(self):
        assert _extract_score("6.0") == 5.0

    def test_clamp_low(self):
        assert _extract_score("-1.0") == 0.0

    def test_zero(self):
        assert _extract_score("0.0") == 0.0

    def test_five(self):
        assert _extract_score("5.0") == 5.0

    def test_no_number(self):
        assert _extract_score("no score here") == -1.0

    def test_empty(self):
        assert _extract_score("") == -1.0

    def test_multiple_numbers_picks_valid(self):
        # "I'd rate this 3.2 out of 5" -> should find 3.2 (valid 0-5 range)
        score = _extract_score("I'd rate this 3.2 out of 5")
        assert 3.0 <= score <= 5.0


class TestAccuracyCheck:

    def test_exact_match(self):
        assert accuracy_check("3.5", "3.5") is True

    def test_within_tolerance(self):
        assert accuracy_check("3.0", "3.8") is True

    def test_outside_tolerance(self):
        assert accuracy_check("1.0", "4.0") is False

    def test_at_boundary(self):
        assert accuracy_check("2.0", "3.0") is True

    def test_just_outside_boundary(self):
        assert accuracy_check("1.9", "3.0") is False

    def test_verbose_response(self):
        assert accuracy_check("The similarity is approximately 4.2", "4.0") is True

    def test_empty_prediction(self):
        assert accuracy_check("", "3.0") is False

    def test_invalid_reference(self):
        assert accuracy_check("3.0", "invalid") is False


class TestLoadSamples:

    def test_load_toy(self):
        samples = load_samples({"dataset": {"source": "toy", "n_samples": 5}})
        assert len(samples) == 5
        assert all(s.sentence1 for s in samples)
        assert all(s.sentence2 for s in samples)

    def test_load_toy_all(self):
        samples = load_samples({"dataset": {"source": "toy", "n_samples": 10}})
        assert len(samples) == 10

    def test_references_are_valid_scores(self):
        samples = load_samples({"dataset": {"source": "toy", "n_samples": 10}})
        for s in samples:
            score = float(s.reference)
            assert 0.0 <= score <= 5.0, f"Score {score} out of range for {s.sid}"
