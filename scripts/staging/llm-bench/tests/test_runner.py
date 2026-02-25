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

"""Unit tests for runner.py (config validation, factory, helpers)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from runner import validate_config, json_safe, _aggregate_tokens


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_valid_config(self):
        cfg = {"name": "math", "dataset": {"source": "gsm8k", "n_samples": 10}}
        validate_config(cfg)  # should not raise

    def test_missing_name(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validate_config({"dataset": {"source": "gsm8k"}})

    def test_invalid_workload(self):
        with pytest.raises(ValueError, match="Unknown workload"):
            validate_config({"name": "nonexistent"})

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples"):
            validate_config({"name": "math", "dataset": {"n_samples": -1}})

    def test_zero_n_samples(self):
        with pytest.raises(ValueError, match="n_samples"):
            validate_config({"name": "math", "dataset": {"n_samples": 0}})

    def test_all_valid_workloads(self):
        for name in ["math", "summarization", "reasoning", "json_extraction", "embeddings"]:
            validate_config({"name": name})  # should not raise


# ---------------------------------------------------------------------------
# json_safe
# ---------------------------------------------------------------------------

class TestJsonSafe:
    def test_primitives(self):
        assert json_safe("hello") == "hello"
        assert json_safe(42) == 42
        assert json_safe(3.14) == 3.14
        assert json_safe(True) is True
        assert json_safe(None) is None

    def test_dict(self):
        assert json_safe({"a": 1, "b": "c"}) == {"a": 1, "b": "c"}

    def test_list(self):
        assert json_safe([1, "two", 3.0]) == [1, "two", 3.0]

    def test_nested(self):
        result = json_safe({"a": [1, {"b": 2}]})
        assert result == {"a": [1, {"b": 2}]}

    def test_non_serializable(self):
        result = json_safe(set([1, 2, 3]))
        assert isinstance(result, str)

    def test_numeric_dict_keys(self):
        result = json_safe({1: "a", 2: "b"})
        assert result == {"1": "a", "2": "b"}


# ---------------------------------------------------------------------------
# _aggregate_tokens
# ---------------------------------------------------------------------------

class TestAggregateTokens:
    def test_with_usage(self):
        outputs = [
            {"extra": {"usage": {"input_tokens": 10, "output_tokens": 20}}},
            {"extra": {"usage": {"input_tokens": 15, "output_tokens": 25}}},
        ]
        total_in, total_out = _aggregate_tokens(outputs)
        assert total_in == 25
        assert total_out == 45

    def test_no_usage(self):
        outputs = [{"extra": {}}, {"extra": {}}]
        total_in, total_out = _aggregate_tokens(outputs)
        assert total_in is None
        assert total_out is None

    def test_partial_usage(self):
        outputs = [
            {"extra": {"usage": {"input_tokens": 10, "output_tokens": 20}}},
            {"extra": {}},
        ]
        total_in, total_out = _aggregate_tokens(outputs)
        assert total_in == 10
        assert total_out == 20

    def test_empty_outputs(self):
        total_in, total_out = _aggregate_tokens([])
        assert total_in is None
        assert total_out is None
