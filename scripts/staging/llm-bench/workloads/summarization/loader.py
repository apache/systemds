import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from datasets import load_dataset


@dataclass
class Sample:
    sid: str
    text: str
    reference: str

TOY_TEXTS = [
    "Large language models (LLMs) are widely used in modern applications. They can generate text, summarize documents, and answer questions.",
    "SystemDS is a machine learning system designed for flexible and scalable analytics. It supports declarative ML programming and optimization.",
    "Benchmarking inference systems involves measuring latency, throughput, and quality across tasks and models under controlled conditions.",
    "Speculative decoding is a technique to accelerate autoregressive generation by using a smaller draft model and verifying with a larger model.",
    "Reproducible experiments require fixed seeds, versioned configs, and consistent environments across runs.",
    "A good benchmark suite includes diverse workloads such as summarization, question answering, and reasoning tasks.",
    "Local inference can reduce cost and improve privacy, but may be limited by hardware constraints and model support.",
    "Hosted APIs offer strong model quality and easy scaling, but introduce network latency and variable cost per token.",
    "Throughput is typically measured in requests per second or tokens per second, depending on the benchmark design.",
    "Accuracy for summarization can be approximated with overlap metrics, but human evaluation is often the gold standard.",
]


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "toy")
    n = int(dataset_cfg.get("n_samples", 10))

    if source == "toy":
        return _load_toy_samples(n)
    elif source == "cnn":
        return _load_cnn_samples(n)
    elif source == "xsum":
        return _load_xsum_samples(n)
    else:
        raise ValueError(f"summarization supports source: toy, cnn, xsum. Got: {source}")


def _load_toy_samples(n: int) -> List[Sample]:
    texts = TOY_TEXTS[: max(1, min(n, len(TOY_TEXTS)))]
    return [Sample(sid=f"toy-{i}", text=t, reference=t) for i, t in enumerate(texts)]


def _load_cnn_samples(n: int) -> List[Sample]:
    try:
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Warning: CNN/DailyMail download failed ({e}), using toy dataset")
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
        print(f"Warning: XSum download failed ({e}), using toy dataset")
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


def accuracy_check(prediction: str, reference: str) -> bool:
    """Quality gate, not a true accuracy metric.

    Checks whether the output looks like a plausible summary:
    non-empty, reasonable length, contains at least one reference term,
    and has sentence structure.  Does NOT measure factual correctness
    or semantic similarity (use ROUGE/BERTScore for that).
    """
    if not prediction or not reference:
        return False

    prediction = prediction.strip()
    reference = reference.strip()

    if len(prediction) < 20:
        return False

    if len(prediction) > max(len(reference) * 5, 500):
        return False

    ref_terms = _tokenize(reference)
    pred_terms = _tokenize(prediction)

    if ref_terms and len(ref_terms) >= 5 and len(ref_terms & pred_terms) == 0:
        return False

    if len(prediction) > 50 and not re.search(r'[.!?]', prediction):
        return False

    if len(pred_terms) < 3:
        return False

    return True
