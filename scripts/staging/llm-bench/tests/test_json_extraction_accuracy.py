#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

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
    _compute_entity_metrics,
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
# _compute_entity_metrics (NER)
# ---------------------------------------------------------------------------

class TestEntityMetrics:
    def test_perfect_match(self):
        ref = {"persons": ["John Smith"], "organizations": ["Google"]}
        pred = {"persons": ["John Smith"], "organizations": ["Google"]}
        m = _compute_entity_metrics(pred, ref)
        assert m["entity_f1"] == pytest.approx(1.0)
        assert m["entities_correct"] == 2

    def test_partial_match(self):
        ref = {"persons": ["John", "Jane"], "organizations": ["Google"]}
        pred = {"persons": ["John"], "organizations": ["Google"]}
        m = _compute_entity_metrics(pred, ref)
        assert m["entities_correct"] == 2
        assert m["entities_reference"] == 3
        assert m["entity_recall"] == pytest.approx(2.0 / 3.0)

    def test_no_match(self):
        ref = {"persons": ["John"]}
        pred = {"persons": ["Bob"]}
        m = _compute_entity_metrics(pred, ref)
        assert m["entity_f1"] == 0.0

    def test_empty_prediction(self):
        ref = {"persons": ["John"]}
        pred = {"persons": []}
        m = _compute_entity_metrics(pred, ref)
        assert m["entity_precision"] == 0.0
        assert m["entity_recall"] == 0.0

    def test_extra_predictions(self):
        ref = {"persons": ["John"]}
        pred = {"persons": ["John", "Jane", "Bob"]}
        m = _compute_entity_metrics(pred, ref)
        assert m["entity_precision"] == pytest.approx(1.0 / 3.0)
        assert m["entity_recall"] == 1.0

    def test_non_list_field_ignored(self):
        ref = {"count": 5, "persons": ["John"]}
        pred = {"count": 5, "persons": ["John"]}
        m = _compute_entity_metrics(pred, ref)
        assert m["entities_reference"] == 1  # only list fields counted


# ---------------------------------------------------------------------------
# NER accuracy_check
# ---------------------------------------------------------------------------

class TestNerAccuracyCheck:
    def test_ner_pass(self):
        ref = json.dumps({"persons": ["John Smith"], "organizations": ["Google"]})
        pred = '{"persons": ["John Smith"], "organizations": ["Google"]}'
        assert accuracy_check(pred, ref) is True

    def test_ner_fail_low_f1(self):
        ref = json.dumps({"persons": ["John", "Jane", "Bob"], "organizations": ["Google", "Apple"]})
        pred = '{"persons": ["Alice"], "organizations": []}'
        assert accuracy_check(pred, ref) is False

    def test_ner_f1_exactly_half(self):
        ref = json.dumps({"persons": ["John", "Jane"]})
        pred = '{"persons": ["John"]}'
        # precision=1.0, recall=0.5, F1=0.667 >= 0.5 -> pass
        assert accuracy_check(pred, ref) is True


# ---------------------------------------------------------------------------
# accuracy_check (scalar path)
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
# load_samples
# ---------------------------------------------------------------------------

class TestLoadSamples:
    def test_invalid_source(self):
        cfg = {"name": "json_extraction", "dataset": {"source": "invalid_source", "n_samples": 5}}
        with pytest.raises(ValueError, match="json_extraction supports source"):
            load_samples(cfg)
