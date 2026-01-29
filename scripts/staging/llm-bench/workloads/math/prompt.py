from typing import Any, Dict
from .loader import Sample


def make_prompt(sample: Sample, cfg: Dict[str, Any]) -> str:
    return (
        "Solve this math problem step-by-step. Show your work and give the final numerical answer.\n\n"
        f"Problem: {sample.question}\n"
    )
