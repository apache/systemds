from typing import Any, Dict
from .loader import Sample


def make_prompt(sample: Sample, cfg: Dict[str, Any]) -> str:
    return (
        "Summarize the following text in 1 sentence, keeping only the key point. "
        "Be concise and shorter than the original.\n\n"
        f"{sample.text}\n"
    )
