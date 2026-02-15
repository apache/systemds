"""MLX backend -- Apple Silicon local inference via mlx-lm."""

import time
from typing import Any, Dict, List

from mlx_lm import load, generate


class MLXBackend:

    def __init__(self, model: str):
        try:
            self.model, self.tokenizer = load(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load MLX model '{model}': {e!r}") from e

    def generate(self, prompts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        max_tokens = int(config.get("max_tokens", config.get("max_output_tokens", 128)))
        temperature = float(config.get("temperature", 0.0))
        results = []

        for p in prompts:
            try:
                t0 = time.perf_counter()
                out = generate(self.model, self.tokenizer, p,
                               max_tokens=max_tokens, temp=temperature, verbose=False)
                t1 = time.perf_counter()

                total_ms = (t1 - t0) * 1000.0

                extra = {}
                try:
                    in_tok = len(self.tokenizer.encode(p))
                    out_tok = len(self.tokenizer.encode(out))
                    extra["usage"] = {
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "total_tokens": in_tok + out_tok,
                    }
                except Exception:
                    pass

                results.append({
                    "text": out,
                    "latency_ms": total_ms,
                    "extra": extra,
                })

            except Exception as e:
                results.append({"text": "", "latency_ms": 0.0, "extra": {"error": repr(e)}})

        return results
