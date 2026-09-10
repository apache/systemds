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
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    sid: str
    sentence1: str
    sentence2: str
    reference: str  # similarity score as string (0.0-5.0)


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "stsb")
    n = int(dataset_cfg.get("n_samples", 10))

    if source == "stsb":
        samples = _load_stsb_samples(n)
    else:
        raise ValueError(f"embeddings supports source: stsb. Got: {source}")

    if len(samples) < n:
        logger.warning("Requested %d samples but only %d available (source=%s)", n, len(samples), source)
    return samples


def _load_stsb_samples(n: int) -> List[Sample]:
    """Load STS-Benchmark from HuggingFace."""
    try:
        dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
    except Exception as e:
        raise RuntimeError(
            f"Could not load STS-B dataset from HuggingFace: {e}. "
            f"Check your internet connection or install the dataset manually."
        ) from e

    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        score = item.get("score", item.get("similarity_score", 0.0))
        s1 = item.get("sentence1", item.get("text1", ""))
        s2 = item.get("sentence2", item.get("text2", ""))
        if not s1 or not s2:
            continue
        samples.append(Sample(
            sid=f"stsb-{i}",
            sentence1=s1,
            sentence2=s2,
            reference=f"{score:.2f}",
        ))
    return samples


def _extract_score(text: str) -> float:
    """Extract a numeric score (0.0-5.0) from model response.
    Returns -1.0 if no valid score found or if score is outside 0-5 range."""
    text = text.strip()
    # try direct float parse first
    try:
        val = float(text)
        if 0.0 <= val <= 5.0:
            return val
        return -1.0  # out of range = extraction failure
    except ValueError:
        pass
    # pick first valid 0-5 number (avoids grabbing "5" from "3.2 out of 5")
    matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    for m in matches:
        val = float(m)
        if 0.0 <= val <= 5.0:
            return val
    return -1.0


def accuracy_check(prediction: str, reference: str) -> bool:
    """Pass if predicted score is within 1.0 of reference (0-5 scale)."""
    pred_score = _extract_score(prediction)
    accuracy_check.last_pred_score = pred_score if pred_score >= 0 else None
    if pred_score < 0:
        return False
    try:
        ref_score = float(reference)
    except ValueError:
        return False
    # within 1.0 point on 0-5 scale
    return abs(pred_score - ref_score) <= 1.0


accuracy_check.last_pred_score = None
