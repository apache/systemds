import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    sid: str
    text: str
    reference: str

TOY_SAMPLES = [
    {
        "text": "Large language models (LLMs) are widely used in modern applications. They can generate text, summarize documents, and answer questions.",
        "reference": "LLMs are versatile tools used for text generation, summarization, and question answering.",
    },
    {
        "text": "SystemDS is a machine learning system designed for flexible and scalable analytics. It supports declarative ML programming and optimization.",
        "reference": "SystemDS enables flexible, scalable machine learning through declarative programming and optimization.",
    },
    {
        "text": "Benchmarking inference systems involves measuring latency, throughput, and quality across tasks and models under controlled conditions.",
        "reference": "Inference benchmarking measures latency, throughput, and quality under controlled conditions.",
    },
    {
        "text": "Speculative decoding is a technique to accelerate autoregressive generation by using a smaller draft model and verifying with a larger model.",
        "reference": "Speculative decoding speeds up text generation by drafting with a small model and verifying with a large one.",
    },
    {
        "text": "Reproducible experiments require fixed seeds, versioned configs, and consistent environments across runs.",
        "reference": "Experiment reproducibility depends on fixed seeds, versioned configs, and consistent environments.",
    },
    {
        "text": "A good benchmark suite includes diverse workloads such as summarization, question answering, and reasoning tasks.",
        "reference": "Effective benchmarks cover diverse workloads including summarization, QA, and reasoning.",
    },
    {
        "text": "Local inference can reduce cost and improve privacy, but may be limited by hardware constraints and model support.",
        "reference": "Local inference offers cost and privacy benefits but faces hardware and model limitations.",
    },
    {
        "text": "Hosted APIs offer strong model quality and easy scaling, but introduce network latency and variable cost per token.",
        "reference": "Cloud APIs provide quality and scalability at the cost of network latency and per-token pricing.",
    },
    {
        "text": "Throughput is typically measured in requests per second or tokens per second, depending on the benchmark design.",
        "reference": "Throughput metrics include requests per second and tokens per second, depending on benchmark design.",
    },
    {
        "text": "Accuracy for summarization can be approximated with overlap metrics, but human evaluation is often the gold standard.",
        "reference": "Summarization accuracy uses overlap metrics as a proxy, though human evaluation remains the gold standard.",
    },
]


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "toy")
    n = int(dataset_cfg.get("n_samples", 10))

    if source == "toy":
        samples = _load_toy_samples(n)
    elif source == "cnn":
        samples = _load_cnn_samples(n)
    elif source == "xsum":
        samples = _load_xsum_samples(n)
    else:
        raise ValueError(f"summarization supports source: toy, cnn, xsum. Got: {source}")

    if len(samples) < n:
        logger.warning("Requested %d samples but only %d available (source=%s)", n, len(samples), source)
    return samples


def _load_toy_samples(n: int) -> List[Sample]:
    items = TOY_SAMPLES[: max(1, min(n, len(TOY_SAMPLES)))]
    return [Sample(sid=f"toy-{i}", text=s["text"], reference=s["reference"]) for i, s in enumerate(items)]


def _load_cnn_samples(n: int) -> List[Sample]:
    try:
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    except Exception as e:
        logger.warning("CNN/DailyMail download failed (%s), using toy dataset", e)
        return _load_toy_samples(n)

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
        dataset = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)
    except Exception as e:
        logger.warning("XSum download failed (%s), using toy dataset", e)
        return _load_toy_samples(n)

    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        document = item["document"]
        if len(document) > 2000:
            continue
        samples.append(Sample(sid=f"xsum-{i}", text=document, reference=item["summary"]))
    return samples


def _tokenize(text: str) -> Set[str]:
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                  'it', 'this', 'that', 'they', 'can', 'may', 'by', 'as'}
    return set(w for w in words if w not in stop_words and len(w) > 2)


def _compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Uses the ``rouge-score`` package if available; otherwise falls back to
    a simple unigram-overlap implementation so the benchmark still works
    without the optional dependency.
    """
    try:
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
    except ImportError:
        logger.debug("rouge-score not installed, using fallback unigram overlap")
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        if not ref_tokens or not pred_tokens:
            return {"rouge1_f": 0.0, "rouge1_p": 0.0, "rouge1_r": 0.0}
        overlap = pred_tokens & ref_tokens
        precision = len(overlap) / len(pred_tokens)
        recall = len(overlap) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"rouge1_f": f1, "rouge1_p": precision, "rouge1_r": recall}


def accuracy_check(prediction: str, reference: str) -> bool:
    """ROUGE-based accuracy check for summarization.

    A prediction passes if its ROUGE-1 F1 score is >= 0.2 (indicating
    meaningful overlap with the reference).  This replaces the previous
    quality-gate heuristic with an actual overlap metric.

    The ROUGE scores are also stored on the function for retrieval by
    the runner (via the ``last_rouge_scores`` attribute).
    """
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
