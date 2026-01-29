"""

Installation:
    1. Download Ollama from https://ollama.ai
    2. Run: ollama pull llama3.2  (or any other model)
    3. Ollama runs as a local server on http://localhost:11434

"""

import time
from typing import Any, Dict, List

import requests


class OllamaBackend:
    """Backend for Ollama local LLM inference."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama back.
        
        Args:
            model: Model name (e.g., "llama3.2", "mistral", "phi3")
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        
        # Verify connection
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available_models = [m["name"] for m in resp.json().get("models", [])]
            
            model_base = model.split(":")[0]
            if not any(model_base in m for m in available_models):
                print(f"Warning: Model '{model}' not found. Available: {available_models}")
                print(f"Run: ollama pull {model}")
                
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (https://ollama.ai)"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama backend: {e}")
    
    def generate(self, prompts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate completions for a list of prompts.
        
        Args:
            prompts: List of prompt strings
            config: Generation config (max_tokens, temperature, etc.)
        
        Returns:
            List of result dicts with text, latency_ms, ttft_ms, etc.
        """
        max_tokens = int(config.get("max_tokens", 512))
        temperature = float(config.get("temperature", 0.0))
        
        results = []
        
        for prompt in prompts:
            try:
                result = self._generate_single(prompt, max_tokens, temperature)
                results.append(result)
            except Exception as e:
                results.append({
                    "text": "",
                    "latency_ms": 0.0,
                    "ttft_ms": 0.0,
                    "generation_ms": 0.0,
                    "extra": {"error": repr(e)}
                })
        
        return results
    
    def _generate_single(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Generate completion for a single prompt with streaming."""
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        
        t0 = time.perf_counter()
        t_first = None
        chunks = []
        
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if not line:
                    continue
                
                import json
                chunk = json.loads(line)
                
                # capture time to first token
                if t_first is None and chunk.get("response"):
                    t_first = time.perf_counter()
                
                if chunk.get("response"):
                    chunks.append(chunk["response"])
                
                if chunk.get("done"):
                    break
        
        t1 = time.perf_counter()
        
        text = "".join(chunks)
        
        total_latency_ms = (t1 - t0) * 1000.0
        ttft_ms = (t_first - t0) * 1000.0 if t_first else total_latency_ms
        generation_ms = (t1 - t_first) * 1000.0 if t_first else 0.0
        
        # estimate token counts (Ollama doesn't always return this)
        # rough estimate: ~4 chars per token
        in_tokens = len(prompt) // 4
        out_tokens = len(text) // 4
        
        # estimate compute cost based on typical consumer GPU (~$0.30/hr equivalent)
        compute_hours = total_latency_ms / 1000.0 / 3600.0
        
        return {
            "text": text,
            "latency_ms": total_latency_ms,
            "ttft_ms": ttft_ms,
            "generation_ms": generation_ms,
            "extra": {
                "usage": {
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens,
                },
                "cost_usd": compute_hours * 0.30,
                "cost_note": "estimated_compute"
            }
        }
    
    def _generate_single_non_streaming(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Generate completion without streaming (simpler but no TTFT)."""
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        t1 = time.perf_counter()
        
        data = resp.json()
        text = data.get("response", "")
        
        total_latency_ms = (t1 - t0) * 1000.0
        
        # get token counts if available
        in_tokens = data.get("prompt_eval_count", len(prompt) // 4)
        out_tokens = data.get("eval_count", len(text) // 4)
        
        # estimate compute cost based on typical consumer GPU (~$0.30/hr equivalent)
        compute_hours = total_latency_ms / 1000.0 / 3600.0
        
        return {
            "text": text,
            "latency_ms": total_latency_ms,
            "ttft_ms": total_latency_ms * 0.1,  # Estimate
            "generation_ms": total_latency_ms * 0.9,
            "extra": {
                "usage": {
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens,
                },
                "cost_usd": compute_hours * 0.30,
                "cost_note": "estimated_compute"
            }
        }
