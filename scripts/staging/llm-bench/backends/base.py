from typing import Any, Dict, List, Optional, Protocol, TypedDict


class GenerationResult(TypedDict, total=False):
    text: str
    latency_ms: float
    ttft_ms: float
    generation_ms: float
    extra: Dict[str, Any]


class InferenceBackend(Protocol):

    def generate(
        self,
        prompts: List[str],
        config: Dict[str, Any],
    ) -> List[GenerationResult]:
        ...
