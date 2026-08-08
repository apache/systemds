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

"""Unit tests for math workload accuracy checking and number extraction."""

import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from workloads.math.loader import (
    accuracy_check,
    extract_number_from_response,
    normalize_number,
    _extract_gsm8k_answer,
    load_samples,
)


# ---------------------------------------------------------------------------
# extract_number_from_response
# ---------------------------------------------------------------------------

class TestExtractNumber:
    def test_explicit_answer_marker(self):
        assert extract_number_from_response("The answer is 42") == "42"

    def test_hash_marker(self):
        assert extract_number_from_response("#### 123") == "123"

    def test_bold_marker(self):
        assert extract_number_from_response("So the result is **75**") == "75"

    def test_boxed(self):
        assert extract_number_from_response("\\boxed{99}") == "99"

    def test_boxed_with_latex_text(self):
        text = "**Final answer:**\n\\[\n\\boxed{25 \\text{ miles}}\n\\]"
        assert extract_number_from_response(text) == "25"

    def test_equals_at_end(self):
        assert extract_number_from_response("5 + 3 = 8") == "8"

    def test_currency(self):
        assert extract_number_from_response("The total profit is $150.") == "150"

    def test_comma_separated_number(self):
        result = extract_number_from_response("The answer is 1,234")
        assert result == "1234"

    def test_no_answer_marker_returns_none(self):
        assert extract_number_from_response("Some text 7 more text 13") is None

    def test_empty_string(self):
        assert extract_number_from_response("") is None

    def test_no_number(self):
        assert extract_number_from_response("no numbers here") is None

    def test_filters_followup(self):
        text = "The answer is 42.\nFollow-up: What is 5 + 3? The answer is 8."
        assert extract_number_from_response(text) == "42"

    def test_decimal_number(self):
        assert extract_number_from_response("The answer is 3.14") == "3.14"

    def test_final_answer_is_pattern(self):
        text = "Step 1: 10 + 5 = 15\nStep 2: 15 * 2 = 30\nThe final answer is 30."
        assert extract_number_from_response(text) == "30"


# ---------------------------------------------------------------------------
# normalize_number
# ---------------------------------------------------------------------------

class TestNormalizeNumber:
    def test_integer(self):
        assert normalize_number("42") == 42.0

    def test_float(self):
        assert normalize_number("3.14") == pytest.approx(3.14)

    def test_comma(self):
        assert normalize_number("1,000") == 1000.0

    def test_empty(self):
        assert normalize_number("") is None

    def test_none(self):
        assert normalize_number(None) is None

    def test_invalid(self):
        assert normalize_number("abc") is None


# ---------------------------------------------------------------------------
# accuracy_check
# ---------------------------------------------------------------------------

class TestMathAccuracyCheck:
    def test_correct_answer(self):
        assert accuracy_check("The answer is 42", "42") is True

    def test_wrong_answer(self):
        assert accuracy_check("The answer is 99", "42") is False

    def test_empty_prediction(self):
        assert accuracy_check("", "42") is False

    def test_empty_reference(self):
        assert accuracy_check("42", "") is False

    def test_verbose_correct(self):
        text = "Let me solve this step by step.\n5 + 3 = 8\n10 * 8 = 80\nThe answer is 80."
        assert accuracy_check(text, "80") is True

    def test_float_match(self):
        assert accuracy_check("The answer is 3.14", "3.14") is True

    def test_float_mismatch(self):
        assert accuracy_check("The answer is 3.15", "3.14") is False


# ---------------------------------------------------------------------------
# _extract_gsm8k_answer
# ---------------------------------------------------------------------------

class TestExtractGsm8kAnswer:
    def test_standard_format(self):
        assert _extract_gsm8k_answer("some work\n#### 42") == "42"

    def test_with_comma(self):
        assert _extract_gsm8k_answer("#### 1,234") == "1234"

    def test_no_marker(self):
        assert _extract_gsm8k_answer("just some text") is None


# ---------------------------------------------------------------------------
# load_samples
# ---------------------------------------------------------------------------

class TestLoadSamples:
    def test_invalid_source(self):
        cfg = {"name": "math", "dataset": {"source": "invalid_source", "n_samples": 5}}
        with pytest.raises(ValueError, match="math supports source"):
            load_samples(cfg)
