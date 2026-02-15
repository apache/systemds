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


TOY_DATASET = [
    {"id": "sts-1", "s1": "A man is playing a guitar.", "s2": "A man is playing a flute.", "score": 2.2},
    {"id": "sts-2", "s1": "A woman is dancing.", "s2": "A woman is dancing in the rain.", "score": 3.8},
    {"id": "sts-3", "s1": "The cat sat on the mat.", "s2": "A cat is sitting on a mat.", "score": 4.6},
    {"id": "sts-4", "s1": "A plane is taking off.", "s2": "A dog is catching a ball.", "score": 0.2},
    {"id": "sts-5", "s1": "The stock market crashed today.", "s2": "Financial markets saw major losses.", "score": 4.0},
    {"id": "sts-6", "s1": "A child is riding a horse.", "s2": "A child is riding a bicycle.", "score": 2.4},
    {"id": "sts-7", "s1": "The president gave a speech.", "s2": "The president delivered an address.", "score": 4.5},
    {"id": "sts-8", "s1": "It is raining outside.", "s2": "The weather is sunny and warm.", "score": 0.8},
    {"id": "sts-9", "s1": "Two dogs are playing in the snow.", "s2": "Two dogs play in the snow.", "score": 4.8},
    {"id": "sts-10", "s1": "A person is cooking dinner.", "s2": "Someone is preparing a meal.", "score": 4.2},
]


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "toy")
    n = int(dataset_cfg.get("n_samples", 10))

    if source == "toy":
        samples = _load_toy_samples(n)
    elif source == "stsb":
        samples = _load_stsb_samples(n)
    else:
        raise ValueError(f"embeddings supports source: toy, stsb. Got: {source}")

    if len(samples) < n:
        logger.warning("Requested %d samples but only %d available (source=%s)", n, len(samples), source)
    return samples


def _load_toy_samples(n: int) -> List[Sample]:
    items = TOY_DATASET[:min(n, len(TOY_DATASET))]
    return [Sample(sid=item["id"], sentence1=item["s1"], sentence2=item["s2"],
                   reference=str(item["score"]))
            for item in items]


def _load_stsb_samples(n: int) -> List[Sample]:
    """Load STS-Benchmark from HuggingFace. Falls back to toy if download fails."""
    try:
        dataset = load_dataset("mteb/stsbenchmark-sts", split="test", trust_remote_code=True)
    except Exception as e:
        logger.warning("STS-B download failed (%s), using toy dataset", e)
        return _load_toy_samples(n)

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
    """Extract a numeric score (0.0-5.0) from model response."""
    text = text.strip()
    # try direct float parse first
    try:
        val = float(text)
        return max(0.0, min(5.0, val))
    except ValueError:
        pass
    # find a decimal number in the response
    matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    for m in reversed(matches):
        val = float(m)
        if 0.0 <= val <= 5.0:
            return val
    # fallback: try first number even if > 5, clamp it
    if matches:
        return max(0.0, min(5.0, float(matches[-1])))
    return -1.0


def accuracy_check(prediction: str, reference: str) -> bool:
    """Check if predicted similarity score is within 1.0 of reference.

    A tolerance of 1.0 on a 0-5 scale (20%) is standard for STS evaluation.
    For finer analysis, Pearson/Spearman correlation is computed in the runner
    via the stored scores.
    """
    pred_score = _extract_score(prediction)
    if pred_score < 0:
        return False
    try:
        ref_score = float(reference)
    except ValueError:
        return False
    # within 1.0 point on 0-5 scale
    return abs(pred_score - ref_score) <= 1.0


# Store last predicted score for correlation computation
accuracy_check.last_pred_score = None


def accuracy_check_with_score(prediction: str, reference: str) -> bool:
    """Same as accuracy_check but also stores the predicted score."""
    pred_score = _extract_score(prediction)
    accuracy_check.last_pred_score = pred_score if pred_score >= 0 else None
    return accuracy_check(prediction, reference)
