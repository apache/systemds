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

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    sid: str
    text: str
    reference: str


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "xsum")
    n = int(dataset_cfg.get("n_samples", 10))

    if source == "cnn":
        samples = _load_cnn_samples(n)
    elif source == "xsum":
        samples = _load_xsum_samples(n)
    else:
        raise ValueError(f"summarization supports source: cnn, xsum. Got: {source}")

    if len(samples) < n:
        logger.warning("Requested %d samples but only %d available (source=%s)", n, len(samples), source)
    return samples



def _load_cnn_samples(n: int) -> List[Sample]:
    try:
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    except Exception as e:
        raise RuntimeError(
            f"Could not load CNN/DailyMail dataset from HuggingFace: {e}. "
            f"Check your internet connection or install the dataset manually."
        ) from e

    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        article = item["article"]
        if len(article) > 2000:
            continue
        samples.append(Sample(sid=f"cnn-{i}", text=article, reference=item["highlights"]))
    return samples


def _load_xsum_samples(n: int) -> List[Sample]:
    try:
        dataset = load_dataset("EdinburghNLP/xsum", split="test")
    except Exception as e:
        raise RuntimeError(
            f"Could not load XSum dataset from HuggingFace: {e}. "
            f"Check your internet connection or install the dataset manually."
        ) from e

    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        document = item["document"]
        if len(document) > 2000:
            continue
        samples.append(Sample(sid=f"xsum-{i}", text=document, reference=item["summary"]))
    return samples


def _compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """ROUGE scores. Requires rouge-score package (listed in requirements.txt)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge1_p": scores["rouge1"].precision,
        "rouge1_r": scores["rouge1"].recall,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rouge2_p": scores["rouge2"].precision,
        "rouge2_r": scores["rouge2"].recall,
        "rougeL_f": scores["rougeL"].fmeasure,
        "rougeL_p": scores["rougeL"].precision,
        "rougeL_r": scores["rougeL"].recall,
    }


def accuracy_check(prediction: str, reference: str) -> bool:
    """Pass if ROUGE-1 F1 >= 0.2. Stores scores in last_rouge_scores."""
    if not prediction or not reference:
        accuracy_check.last_rouge_scores = {}
        return False

    prediction = prediction.strip()
    reference = reference.strip()

    if len(prediction) < 10:
        accuracy_check.last_rouge_scores = {}
        return False

    scores = _compute_rouge(prediction, reference)
    accuracy_check.last_rouge_scores = scores

    return scores.get("rouge1_f", 0.0) >= 0.2

accuracy_check.last_rouge_scores = {}
