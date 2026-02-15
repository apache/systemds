from typing import Any, Dict
from .loader import Sample


def make_prompt(sample: Sample, cfg: Dict[str, Any]) -> str:
    return (
        "Rate the semantic similarity between these two sentences on a scale "
        "from 0.0 (completely unrelated) to 5.0 (identical meaning).\n\n"
        f"Sentence 1: {sample.sentence1}\n"
        f"Sentence 2: {sample.sentence2}\n\n"
        "Output only the numeric score (e.g., 3.5). Do not explain."
    )
