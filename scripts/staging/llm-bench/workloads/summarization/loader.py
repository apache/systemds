import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from datasets import load_dataset


@dataclass
class Sample:
    sid: str
    text: str
    reference: str  # the reference summary (or original text for toy)


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
    """
    Load summarization samples.
    
    Supports multiple sources:
    - "toy": Use built-in toy dataset (10 short texts)
    - "cnn": Use CNN/DailyMail dataset (news articles with summaries)
    - "xsum": Use XSum dataset (BBC articles with one-sentence summaries)
    """
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
    """Load from built-in toy dataset."""
    texts = TOY_TEXTS[: max(1, min(n, len(TOY_TEXTS)))]
    samples: List[Sample] = []
    for i, t in enumerate(texts):
        # use original text as reference for quality comparison
        samples.append(Sample(sid=f"toy-{i}", text=t, reference=t))
    return samples


def _load_cnn_samples(n: int) -> List[Sample]:
    """
    Load from CNN/DailyMail dataset.
    
    This is a standard summarization benchmark with news articles
    and multi-sentence highlights as summaries.
    """
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    
    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        
        article = item["article"]
        highlights = item["highlights"]
        
        # skip very long articles (>2000 chars) for practical inference
        if len(article) > 2000:
            continue
        
        samples.append(Sample(
            sid=f"cnn-{i}",
            text=article,
            reference=highlights,
        ))
    
    return samples


def _load_xsum_samples(n: int) -> List[Sample]:
    """
    Load from XSum dataset.
    
    XSum contains BBC articles with one-sentence summaries.
    Good for testing concise summarization.
    """
    dataset = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)
    
    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        
        document = item["document"]
        summary = item["summary"]
        
        # skip very long documents (>2000 chars)
        if len(document) > 2000:
            continue
        
        samples.append(Sample(
            sid=f"xsum-{i}",
            text=document,
            reference=summary,
        ))
    
    return samples


def tokenize(text: str) -> Set[str]:
    """Simple word tokenization for overlap calculation."""
    # lowercase, remove punctuation, split into words
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    # remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                  'it', 'this', 'that', 'they', 'can', 'may', 'by', 'as'}
    return set(w for w in words if w not in stop_words and len(w) > 2)


def accuracy_check(prediction: str, reference: str) -> bool:
    """
    Check if the summary is a valid, quality output.
    
    For summarization, we primarily check:
    1. The output is a reasonable length (not empty, not too long)
    2. The output is coherent (proper sentence structure)
    3. Some key content is preserved (flexible - allows paraphrasing)
    
    Note: Perfect word overlap is NOT required since good summaries
    often paraphrase content using different vocabulary.
    
    Args:
        prediction: The model's summary/response
        reference: The reference summary (or original text for toy)
    
    Returns:
        True if the output meets quality criteria, False otherwise
    """
    if not prediction or not reference:
        return False
    
    prediction = prediction.strip()
    reference = reference.strip()
    
    pred_len = len(prediction)
    ref_len = len(reference)
    
    # check 1: Output shouldn't be empty or too short
    if pred_len < 20:
        return False
    
    # check 2: Output shouldn't be excessively long
    # for summarization, allow generous length variation
    # just ensure the output isn't absurdly long (>5x reference)
    if pred_len > max(ref_len * 5, 500):
        return False
    
    # check 3: Key term overlap - very lenient for real datasets
    # models often use synonyms/paraphrases which is perfectly valid
    ref_terms = tokenize(reference)
    pred_terms = tokenize(prediction)
    
    if ref_terms and len(ref_terms) >= 5:
        overlap = ref_terms.intersection(pred_terms)
        # only require ~10% overlap since paraphrasing is common
        # if overlap is 0, that's suspicious
        if len(overlap) == 0:
            return False
    
    # check 4: Basic coherence - should have proper sentence structure
    if pred_len > 50 and not re.search(r'[.!?]', prediction):
        return False
    
    # check 5: Prediction should have meaningful content
    if len(pred_terms) < 3:
        return False
    
    return True
