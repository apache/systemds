"""

vLLM is the industry-standard for LLM inference serving, offering:
- High throughput with PagedAttention
- Continuous batching
- OpenAI-compatible API

Installation (requires NVIDIA GPU or specific setup):
    pip install vllm

Running vLLM server:
    # Start vLLM server with a model
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-chat-hf \
        --host 0.0.0.0 --port 8000

    # Or use Docker
    docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
        --model meta-llama/Llama-2-7b-chat-hf
"""

import os
import time
from typing import Any, Dict, List

import requests


class VLLMBackend:
    """
    Backend for vLLM inference server.
    
    vLLM exposes an OpenAI-compatible API, so this backend uses the same
    format as OpenAI but connects to a local/remote vLLM server.
    """
    
    def __init__(self, model: str, base_url: str = None):
        """
        Initialize vLLM backend.
        
        Args:
            model: Model name (must match what vLLM server is running)
            base_url: vLLM server URL (default: http://localhost:8000 or VLLM_BASE_URL env)
        """
        self.model = model
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        self.base_url = self.base_url.rstrip("/")
        

        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            models_data = resp.json()
            available_models = [m["id"] for m in models_data.get("data", [])]
            
            if model not in available_models:
                print(f"Warning: Model '{model}' not found on vLLM server.")
                print(f"Available models: {available_models}")
                print(f"Make sure vLLM is running with: python -m vllm.entrypoints.openai.api_server --model {model}")
                
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.base_url}. "
                f"Start vLLM with: python -m vllm.entrypoints.openai.api_server --model {model}"
            )
        except Exception as e:
            print(f"Warning: Could not verify vLLM server: {e}")
    
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
        
        url = f"{self.base_url}/v1/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        
        t0 = time.perf_counter()
        t_first = None
        chunks = []
        usage_data = None
        
        # stream response
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if not line:
                    continue
                
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                
                data_str = line[6:] 
                if data_str == "[DONE]":
                    break
                
                import json
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                
                # time to first token
                choices = chunk.get("choices", [])
                if choices and t_first is None:
                    text = choices[0].get("text", "")
                    if text:
                        t_first = time.perf_counter()
                
                for choice in choices:
                    text = choice.get("text", "")
                    if text:
                        chunks.append(text)
                

                if "usage" in chunk:
                    usage_data = chunk["usage"]
        
        t1 = time.perf_counter()
        
        # combine response
        text = "".join(chunks)
        
        # metrics
        total_latency_ms = (t1 - t0) * 1000.0
        ttft_ms = (t_first - t0) * 1000.0 if t_first else total_latency_ms * 0.1
        generation_ms = (t1 - t_first) * 1000.0 if t_first else total_latency_ms * 0.9
        
        # token counts
        if usage_data:
            in_tokens = usage_data.get("prompt_tokens", 0)
            out_tokens = usage_data.get("completion_tokens", 0)
        else:
            # estimate
            in_tokens = len(prompt) // 4
            out_tokens = len(text) // 4
        
        # estimate compute cost based on cloud GPU equivalent
        # T4 GPU: ~$0.35/hr, A100: ~$1.50/hr - use T4 as typical Colab GPU

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
                "cost_usd": compute_hours * 0.35,  # T4 GPU equivalent
                "cost_note": "estimated_compute"
            }
        }
    
    def _generate_single_non_streaming(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Generate completion without streaming."""
        
        url = f"{self.base_url}/v1/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        t1 = time.perf_counter()
        
        data = resp.json()
        
        choices = data.get("choices", [])
        text = choices[0].get("text", "") if choices else ""
        
        usage = data.get("usage", {})
        in_tokens = usage.get("prompt_tokens", len(prompt) // 4)
        out_tokens = usage.get("completion_tokens", len(text) // 4)
        
        total_latency_ms = (t1 - t0) * 1000.0
        
        # estimate compute cost based on T4 GPU (~$0.35/hr)
        compute_hours = total_latency_ms / 1000.0 / 3600.0
        
        return {
            "text": text,
            "latency_ms": total_latency_ms,
            "ttft_ms": total_latency_ms * 0.1,
            "generation_ms": total_latency_ms * 0.9,
            "extra": {
                "usage": {
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens,
                },
                "cost_usd": compute_hours * 0.35,
                "cost_note": "estimated_compute"
            }
        }
