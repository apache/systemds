"""Unit tests for JSON extraction workload accuracy checking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
from workloads.json_extraction.loader import (
    accuracy_check,
    extract_json_from_prediction,
    _values_match_strict,
    _normalize_value,
    load_samples,
)


# ---------------------------------------------------------------------------
# extract_json_from_prediction
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json(self):
        result = extract_json_from_prediction('{"name": "John", "age": 35}')
        assert result == {"name": "John", "age": 35}

    def test_json_in_markdown(self):
        text = 'Here is the JSON:\n```json\n{"name": "John"}\n```'
        result = extract_json_from_prediction(text)
        assert result == {"name": "John"}

    def test_json_with_surrounding_text(self):
        text = 'The extracted information is:\n{"city": "Paris"}\nThat is all.'
        result = extract_json_from_prediction(text)
        assert result is not None
        assert result["city"] == "Paris"

    def test_no_json(self):
        assert extract_json_from_prediction("no json here") is None

    def test_empty(self):
        assert extract_json_from_prediction("") is None

    def test_invalid_json(self):
        assert extract_json_from_prediction("{invalid json}") is None


# ---------------------------------------------------------------------------
# _values_match_strict
# ---------------------------------------------------------------------------

class TestValuesMatchStrict:
    def test_exact_string(self):
        assert _values_match_strict("John", "John") is True

    def test_case_insensitive(self):
        assert _values_match_strict("john", "John") is True

    def test_title_variant_dr(self):
        assert _values_match_strict("Dr. Maria Garcia", "Maria Garcia") is True

    def test_title_variant_mr(self):
        assert _values_match_strict("Mr. Smith", "Smith") is True

    def test_different_strings(self):
        assert _values_match_strict("Alice", "Bob") is False

    def test_exact_int(self):
        assert _values_match_strict(35, 35) is True

    def test_int_float_equivalent(self):
        assert _values_match_strict(35.0, 35) is True

    def test_different_numbers(self):
        assert _values_match_strict(35, 36) is False

    def test_boolean_match(self):
        assert _values_match_strict(True, True) is True

    def test_boolean_mismatch(self):
        assert _values_match_strict(True, False) is False


# ---------------------------------------------------------------------------
# accuracy_check
# ---------------------------------------------------------------------------

class TestJsonAccuracyCheck:
    def test_perfect_match(self):
        ref = json.dumps({"name": "John Smith", "age": 35, "city": "San Francisco"})
        pred = '{"name": "John Smith", "age": 35, "city": "San Francisco"}'
        assert accuracy_check(pred, ref) is True

    def test_missing_field(self):
        ref = json.dumps({"name": "John", "age": 35, "city": "SF"})
        pred = '{"name": "John", "age": 35}'
        assert accuracy_check(pred, ref) is False

    def test_wrong_value(self):
        ref = json.dumps({"name": "John", "age": 35})
        pred = '{"name": "Jane", "age": 35}'
        # 50% match (1/2), below 90% threshold
        assert accuracy_check(pred, ref) is False

    def test_no_json_in_prediction(self):
        ref = json.dumps({"name": "John"})
        pred = "I don't know"
        assert accuracy_check(pred, ref) is False

    def test_invalid_reference(self):
        assert accuracy_check('{"a": 1}', "not json") is False

    def test_json_in_markdown(self):
        ref = json.dumps({"name": "John", "age": 35})
        pred = '```json\n{"name": "John", "age": 35}\n```'
        assert accuracy_check(pred, ref) is True

    def test_90_percent_threshold(self):
        # 9/10 fields correct = 90% -> pass
        ref_dict = {f"field_{i}": f"val_{i}" for i in range(10)}
        pred_dict = dict(ref_dict)
        pred_dict["field_9"] = "wrong"  # 1 wrong out of 10
        ref = json.dumps(ref_dict)
        pred = json.dumps(pred_dict)
        assert accuracy_check(pred, ref) is True

    def test_below_threshold(self):
        # 8/10 fields correct = 80% -> fail
        ref_dict = {f"field_{i}": f"val_{i}" for i in range(10)}
        pred_dict = dict(ref_dict)
        pred_dict["field_8"] = "wrong"
        pred_dict["field_9"] = "wrong"
        ref = json.dumps(ref_dict)
        pred = json.dumps(pred_dict)
        assert accuracy_check(pred, ref) is False


# ---------------------------------------------------------------------------
# load_samples (toy)
# ---------------------------------------------------------------------------

class TestLoadSamples:
    def test_load_toy(self):
        cfg = {"name": "json_extraction", "dataset": {"source": "toy", "n_samples": 5}}
        samples = load_samples(cfg)
        assert len(samples) == 5
        assert all(s.text for s in samples)
        assert all(s.reference for s in samples)
        assert all(s.schema for s in samples)

    def test_references_are_valid_json(self):
        cfg = {"name": "json_extraction", "dataset": {"source": "toy", "n_samples": 10}}
        samples = load_samples(cfg)
        for s in samples:
            parsed = json.loads(s.reference)
            assert isinstance(parsed, dict)
