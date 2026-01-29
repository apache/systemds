import time
from typing import Any, Dict, List

import mlx.core as mx
from mlx_lm import load, generate


class MLXBackend:
    def __init__(self, model: str):
        try:
            self.model, self.tokenizer = load(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load MLX model '{model}': {e!r}") from e

    def generate(self, prompts: List[str], config: Dict[str, Any]):
        max_tokens = int(config.get("max_tokens", 128))
        temperature = float(config.get("temperature", 0.0))
        
        results = []
        
        for p in prompts:
            try:
                t0 = time.perf_counter()
                
                out = generate(
                    self.model,
                    self.tokenizer,
                    p,
                    max_tokens=max_tokens,
                    temp=temperature, 
                    verbose=False,
                )
                
                t1 = time.perf_counter()
                
                total_latency_ms = (t1 - t0) * 1000.0
                # estimate TTFT as ~10% of total time (first token overhead)
                ttft_ms = total_latency_ms * 0.1
                generation_ms = total_latency_ms * 0.9
                

                in_tokens = None
                out_tokens = None
                try:
                    in_tokens = len(self.tokenizer.encode(p))
                    out_tokens = len(self.tokenizer.encode(out))
                except Exception:
                    pass
                
                usage = {}
                if in_tokens is not None:
                    usage["input_tokens"] = in_tokens
                if out_tokens is not None:
                    usage["output_tokens"] = out_tokens
                if in_tokens is not None and out_tokens is not None:
                    usage["total_tokens"] = in_tokens + out_tokens
                
                extra = {"usage": usage} if usage else {}
                # estimate cost based on Apple Silicon(~$0.50/hr equivalent)
                # estimate of what similar cloud compute would cost
                compute_hours = total_latency_ms / 1000.0 / 3600.0
                extra["cost_usd"] = compute_hours * 0.50  # ~$0.50/hr for Apple Silicon equivalent
                extra["cost_note"] = "estimated_compute"
                
                results.append({
                    "text": out,
                    "latency_ms": total_latency_ms,
                    "ttft_ms": ttft_ms,
                    "generation_ms": generation_ms,
                    "extra": extra
                })
                
            except Exception as e:
                results.append({
                    "text": "",
                    "latency_ms": 0.0,
                    "ttft_ms": 0.0,  
                    "generation_ms": 0.0,  
                    "extra": {"error": repr(e)}
                })
        
        return results