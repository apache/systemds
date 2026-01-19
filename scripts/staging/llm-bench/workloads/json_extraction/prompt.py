from typing import Any, Dict

from .loader import Sample


def make_prompt(sample: Sample, cfg: Dict[str, Any]) -> str:
    """
    Format a JSON extraction prompt for the model.
    
    Instructs the model to extract structured information from text
    and return valid JSON with specified fields.
    """
    return (
        "You are a JSON extraction assistant. Extract information from the text below.\n"
        "Output ONLY a valid JSON object. Do NOT write code. Do NOT explain.\n"
        "Start your response with { and end with }.\n\n"
        f"Text: {sample.text}\n\n"
        f"Extract these fields: {sample.schema}\n\n"
        "JSON output:"
    )
