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
    puzzle: str
    puzzle_type: str
    reference: str

def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "boolq")
    n = int(dataset_cfg.get("n_samples", 10))

    if source == "logiqa":
        samples = _load_logiqa_samples(n)
    elif source == "boolq":
        samples = _load_boolq_samples(n)
    else:
        raise ValueError(f"reasoning supports source: logiqa, boolq. Got: {source}")

    if len(samples) < n:
        logger.warning("Requested %d samples but only %d available (source=%s)", n, len(samples), source)
    return samples


def _load_logiqa_samples(n: int) -> List[Sample]:
    """LogiQA multiple-choice logical reasoning."""
    try:
        dataset = load_dataset("lucasmccabe/logiqa", split="test")
    except Exception as e:
        raise RuntimeError(
            f"Could not load LogiQA dataset from HuggingFace: {e}. "
            f"Check your internet connection or install the dataset manually."
        ) from e

    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        options_text = "\n".join(f"{chr(65+j)}. {opt}" for j, opt in enumerate(item["options"]))
        puzzle = (f"{item['context']}\n\nQuestion: {item['query']}\n\n"
                  f"Options:\n{options_text}\n\nAnswer with just the letter (A, B, C, or D).")
        samples.append(Sample(sid=f"logiqa-{i}", puzzle=puzzle,
                              puzzle_type="logical_reasoning",
                              reference=chr(65 + item["correct_option"])))
    return samples


def _load_boolq_samples(n: int) -> List[Sample]:
    """BoolQ yes/no reading comprehension."""
    try:
        dataset = load_dataset("google/boolq", split="validation")
    except Exception as e:
        raise RuntimeError(
            f"Could not load BoolQ dataset from HuggingFace: {e}. "
            f"Check your internet connection or install the dataset manually."
        ) from e

    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        puzzle = f"Passage: {item['passage']}\n\nQuestion: {item['question']}\n\nAnswer with just 'Yes' or 'No'."
        samples.append(Sample(sid=f"boolq-{i}", puzzle=puzzle,
                              puzzle_type="boolean_reasoning",
                              reference="Yes" if item["answer"] else "No"))
    return samples


def _normalize(answer: str) -> str:
    answer = answer.lower().strip()
    for prefix in ["the answer is", "answer:", "answer is", "the final answer is",
                   "final answer:", "therefore,", "so,", "thus,"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    return answer.rstrip(".,!?")


def _extract_answer(prediction: str) -> Optional[str]:
    """Extract final answer from model output."""
    prediction = prediction.strip()

    # #### format
    m = re.search(r"####\s*(.+?)$", prediction, re.MULTILINE)
    if m:
        return m.group(1).strip()

    # "answer is X" patterns
    for pat in [r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)",
                r"(?:the\s+)?(?:final\s+)?answer[:\s]+([^\n.]+)",
                r"therefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)",
                r"thus[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)",
                r"so[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)",
                r"conclusion[:\s]+([^\n.]+)"]:
        m = re.search(pat, prediction, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # boxed / bold
    m = re.search(r"\\boxed\{([^}]+)\}", prediction)
    if m:
        return m.group(1).strip()
    m = re.search(r"\*\*([^*]+)\*\*\s*$", prediction, re.MULTILINE)
    if m:
        return m.group(1).strip()

    return None


def _extract_boolean(prediction: str) -> Optional[str]:
    """Extract yes/no from prediction. Takes last standalone match if multiple."""
    text = prediction.strip()
    if not text:
        return None

    # standalone yes/no line (take last)
    found = None
    for line in text.split('\n'):
        word = line.strip().lower().rstrip(".,!?:;")
        if word in ("yes", "no"):
            found = word
    if found is not None:
        return found

    # first word is yes/no
    first_word = text.split()[0].lower().rstrip(".,!?:;")
    if first_word in ("yes", "no"):
        return first_word

    # last word of last line
    for line in reversed(text.split('\n')):
        line = line.strip()
        if line:
            words = re.findall(r'[a-zA-Z]+', line.lower())
            if words and words[-1] in ("yes", "no"):
                return words[-1]
            break  # only check the last non-empty line

    return None


def accuracy_check(prediction: str, reference: str) -> bool:
    ref_n = _normalize(reference)
    is_boolean = ref_n in ("yes", "no")

    # boolean (BoolQ)
    if is_boolean:
        pred_answer = _extract_answer(prediction)
        if pred_answer is not None:
            pred_n = _normalize(pred_answer)
            if pred_n == ref_n:
                return True
            # "clearly no", "definitely yes" -- grab last word
            words = pred_n.split()
            if words and words[-1] in ("yes", "no") and words[-1] == ref_n:
                return True
        boolean_answer = _extract_boolean(prediction)
        if boolean_answer is not None:
            return boolean_answer == ref_n
        return False

    # non-boolean (LogiQA)
    pred_answer = _extract_answer(prediction)

    if pred_answer is None:
        pred_norm = _normalize(prediction)
        return bool(re.search(r'\b' + re.escape(ref_n) + r'\b', pred_norm))

    pred_n = _normalize(pred_answer)

    if pred_n == ref_n:
        return True

    if re.search(r'\b' + re.escape(ref_n) + r'\b', pred_n):
        return True
    if re.search(r'\b' + re.escape(pred_n) + r'\b', ref_n):
        return True

    try:
        pnums = re.findall(r'-?\d+(?:\.\d+)?', pred_n)
        rnums = re.findall(r'-?\d+(?:\.\d+)?', ref_n)
        if pnums and rnums and float(pnums[-1]) == float(rnums[-1]):
            return True
    except (ValueError, IndexError):
        pass

    return False
