import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class Sample:
    sid: str
    puzzle: str
    puzzle_type: str
    reference: str

TOY_DATASET = [
    {"id": "seq-1", "type": "sequence",
     "puzzle": "What comes next in this sequence? 2, 6, 12, 20, 30, ?", "reference": "42"},
    {"id": "seq-2", "type": "sequence",
     "puzzle": "What is the next number in this sequence? 1, 1, 2, 3, 5, 8, 13, ?", "reference": "21"},
    {"id": "seq-3", "type": "sequence",
     "puzzle": "Complete the pattern: 3, 9, 27, 81, ?", "reference": "243"},
    {"id": "pat-1", "type": "pattern",
     "puzzle": "If A=1, B=2, C=3, and so on, what is the sum of the letters in the word 'CAT'?", "reference": "24"},
    {"id": "pat-2", "type": "pattern",
     "puzzle": "In a code, APPLE is written as ELPPA. How would ORANGE be written in the same code?", "reference": "EGNARO"},
    {"id": "ded-1", "type": "deductive",
     "puzzle": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?", "reference": "No"},
    {"id": "ded-2", "type": "deductive",
     "puzzle": "If all doctors are professionals, and all professionals have degrees, what can we conclude about doctors?",
     "reference": "All doctors have degrees"},
    {"id": "ded-3", "type": "deductive",
     "puzzle": "Tom is taller than Jerry. Jerry is taller than Spike. Who is the shortest?", "reference": "Spike"},
    {"id": "math-1", "type": "mathematical",
     "puzzle": "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost in cents?",
     "reference": "5"},
    {"id": "math-2", "type": "mathematical",
     "puzzle": "If 5 machines take 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
     "reference": "5"},
]


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "toy")
    n = int(dataset_cfg.get("n_samples", 10))

    if source == "toy":
        return _load_toy_samples(n)
    elif source == "logiqa":
        return _load_logiqa_samples(n)
    elif source == "boolq":
        return _load_boolq_samples(n)
    else:
        raise ValueError(f"reasoning supports source: toy, logiqa, boolq. Got: {source}")


def _load_toy_samples(n: int) -> List[Sample]:
    return [Sample(sid=item["id"], puzzle=item["puzzle"], puzzle_type=item["type"], reference=item["reference"])
            for item in TOY_DATASET[:n]]


def _load_logiqa_samples(n: int) -> List[Sample]:
    """LogiQA multiple-choice logical reasoning. Falls back to toy if download fails."""
    try:
        dataset = load_dataset("lucasmccabe/logiqa", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Warning: LogiQA download failed ({e}), using toy dataset")
        return _load_toy_samples(n)

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
    """BoolQ yes/no reading comprehension. Falls back to toy if download fails."""
    try:
        dataset = load_dataset("google/boolq", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"Warning: BoolQ download failed ({e}), using toy dataset")
        return _load_toy_samples(n)

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

    # last short standalone line
    for line in reversed(prediction.strip().split('\n')):
        line = line.strip()
        if line and len(line) < 100 and not line.startswith('#'):
            if re.match(r"^[\w\s\-\',]+$", line) or re.match(r"^\d+$", line):
                return line

    return None


def accuracy_check(prediction: str, reference: str) -> bool:
    """Check if extracted answer matches reference (exact, word-boundary, or numeric)."""
    pred_answer = _extract_answer(prediction)

    if pred_answer is None:
        ref_norm = _normalize(reference)
        pred_norm = _normalize(prediction)
        return bool(re.search(r'\b' + re.escape(ref_norm) + r'\b', pred_norm))

    pred_n = _normalize(pred_answer)
    ref_n = _normalize(reference)

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
