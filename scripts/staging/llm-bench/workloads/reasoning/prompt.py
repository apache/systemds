from typing import Any, Dict

from .loader import Sample


def make_prompt(sample: Sample, cfg: Dict[str, Any]) -> str:
    """
    Format a logical reasoning prompt for the model.
    
    Instructs the model to think step-by-step and provide a clear final answer.
    """
    return (
        "Solve this logic puzzle step-by-step. "
        "Show your reasoning clearly, then state your final answer.\n\n"
        f"Puzzle: {sample.puzzle}\n\n"
        "Think through this carefully and give your answer."
    )
