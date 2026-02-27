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
from typing import Any, Dict, List, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    sid: str
    question: str
    reference: str

def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "gsm8k")
    n = int(dataset_cfg.get("n_samples", 10))

    if source == "gsm8k":
        samples = _load_gsm8k_samples(n)
    else:
        raise ValueError(f"math supports source: gsm8k. Got: {source}")

    if len(samples) < n:
        logger.warning("Requested %d samples but only %d available (source=%s)", n, len(samples), source)
    return samples


def _load_gsm8k_samples(n: int) -> List[Sample]:
    """Load GSM8K grade-school math problems."""
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        raise RuntimeError(
            f"Could not load GSM8K dataset from HuggingFace: {e}. "
            f"Check your internet connection or install the dataset manually."
        ) from e

    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        final = _extract_gsm8k_answer(item["answer"])
        if final is not None:
            samples.append(Sample(sid=f"gsm8k-{i}", question=item["question"], reference=final))
    return samples


def _extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """Extract number after '####' in GSM8K answer format."""
    match = re.search(r'####\s*([0-9,.\-]+)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    return None


def extract_number_from_response(text: str) -> Optional[str]:
    """Extract the final numerical answer from model output."""
    if not text:
        return None
    text = text.strip()

    def clean_num(s: str) -> str:
        s = s.replace(',', '').strip()
        if s.endswith('.') and s.count('.') == 1:
            s = s[:-1]
        return s

    main = text
    for marker in [r'\bFollow-up\b', r'\bBonus\b', r'\bExtra\b', r'\bNow\s+try\b',
                   r'\bPractice\b', r'\bExercise\b', r'\bQuestion\s*\d+[:\s]']:
        m = re.search(marker, text, re.IGNORECASE)
        if m:
            main = text[:m.start()]
            break

    # explicit answer markers
    for pat in [r'####\s*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'[Aa]nswer[:\s]+[A-Za-z\s]*\$?([0-9,]+(?:\.[0-9]+)?)',
                r'takes?\s+(\d+)\s+(?:bolts?|cups?|items?|pieces?)\s+(?:in\s+total|total)',
                r'(\d+)\s+(?:bolts?|cups?|items?|pieces?)\s+in\s+total']:
        matches = re.findall(pat, main, re.IGNORECASE)
        if matches:
            return clean_num(matches[0])

    # bold / boxed
    for pat in [r'\*\*\$?([0-9,]+(?:\.[0-9]+)?)[^*]*\*\*',
                r'\\boxed\{([0-9,]+(?:\.[0-9]+)?)\}']:
        matches = re.findall(pat, main, re.IGNORECASE)
        if matches:
            return clean_num(matches[0])

    # '= X' at end of line
    for line in reversed(main.split('\n')[-5:]):
        m = re.search(r'=\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:/day|/week|per\s+\w+)?\s*[.!?]?\s*$',
                       line.strip())
        if m:
            return clean_num(m.group(1))

    # profit / earnings / total
    last_lines = '\n'.join(main.strip().split('\n')[-5:])
    for pat in [r'(?:profit|earnings|total|made|earned|is|are)\s+(?:of\s+)?\$([0-9,]+(?:\.[0-9]+)?)',
                r'\$([0-9,]+(?:\.[0-9]+)?)\s*[.!]?\s*$']:
        matches = re.findall(pat, last_lines, re.IGNORECASE)
        if matches:
            return clean_num(matches[-1])

    # any currency
    currency = re.findall(r'\$([0-9,]+(?:\.[0-9]+)?)', main)
    if currency:
        return clean_num(currency[-1])

    return None


def normalize_number(num_str: str) -> Optional[float]:
    if not num_str:
        return None
    try:
        return float(num_str.replace(',', '').strip())
    except ValueError:
        return None


def accuracy_check(prediction: str, reference: str) -> bool:
    """Exact numerical match between extracted answer and reference."""
    if not prediction or not reference:
        return False
    pred_str = extract_number_from_response(prediction)
    if pred_str is None:
        return False
    pred = normalize_number(pred_str)
    ref = normalize_number(reference)
    if pred is None or ref is None:
        return False
    return abs(pred - ref) < 1e-6
