"""Ollama backend -- connects to a running Ollama server."""

import json
import os
import time
from typing import Any, Dict, List

import requests


class OllamaBackend:

    def __init__(self, model: str, base_url: str = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            if not any(model.split(":")[0] in m for m in available):
                print(f"Warning: '{model}' not found. Available: {available}")
                print(f"Run: ollama pull {model}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}")
        except Exception as e:
            raise RuntimeError(f"Ollama init failed: {e}")

    def generate(self, prompts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        max_tokens = int(config.get("max_tokens", config.get("max_output_tokens", 512)))
        temperature = float(config.get("temperature", 0.0))
        results = []
        for prompt in prompts:
            try:
                results.append(self._generate_single(prompt, max_tokens, temperature))
            except Exception as e:
                results.append({"text": "", "latency_ms": 0.0, "extra": {"error": repr(e)}})
        return results

    def _generate_single(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }

        t0 = time.perf_counter()
        t_first = None
        chunks = []
        done_chunk = None

        with requests.post(f"{self.base_url}/api/generate", json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if t_first is None and chunk.get("response"):
                    t_first = time.perf_counter()
                if chunk.get("response"):
                    chunks.append(chunk["response"])
                if chunk.get("done"):
                    done_chunk = chunk
                    break

        t1 = time.perf_counter()
        text = "".join(chunks)
        total_ms = (t1 - t0) * 1000.0
        ttft_ms = (t_first - t0) * 1000.0 if t_first else total_ms
        gen_ms = (t1 - t_first) * 1000.0 if t_first else 0.0

        # Ollama returns real token counts in the done chunk
        extra: Dict[str, Any] = {}
        if done_chunk:
            in_tok = done_chunk.get("prompt_eval_count")
            out_tok = done_chunk.get("eval_count")
            if in_tok is not None or out_tok is not None:
                usage: Dict[str, Any] = {}
                if in_tok is not None:
                    usage["input_tokens"] = in_tok
                if out_tok is not None:
                    usage["output_tokens"] = out_tok
                if in_tok is not None and out_tok is not None:
                    usage["total_tokens"] = in_tok + out_tok
                extra["usage"] = usage

        return {
            "text": text,
            "latency_ms": total_ms,
            "ttft_ms": ttft_ms,
            "generation_ms": gen_ms,
            "extra": extra,
        }
