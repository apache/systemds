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

"""Unit tests for summarization workload accuracy checking (ROUGE-based)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from workloads.summarization.loader import (
    accuracy_check,
    _compute_rouge,
    load_samples,
)


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
# load_samples
# ---------------------------------------------------------------------------

class TestLoadSamples:
    def test_invalid_source(self):
        cfg = {"name": "summarization", "dataset": {"source": "invalid_source", "n_samples": 5}}
        with pytest.raises(ValueError, match="summarization supports source"):
            load_samples(cfg)
