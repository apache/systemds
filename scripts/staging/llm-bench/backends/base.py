from typing import Any, Dict, List, Optional, Protocol, TypedDict


class GenerationResult(TypedDict, total=False):
    text: str
    latency_ms: float
    tokens: Optional[int]
    extra: Dict[str, Any]


class InferenceBackend(Protocol):
    """
    Minimal contract all inference backends must implement.

    """

    def generate(
        self,
        prompts: List[str],
        config: Dict[str, Any],
    ) -> List[GenerationResult]:
        ...
