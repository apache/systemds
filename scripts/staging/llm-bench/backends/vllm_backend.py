#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

"""vLLM backend -- connects to a running vLLM OpenAI-compatible server."""

import json
import logging
import os
import time
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class VLLMBackend:

    def __init__(self, model: str, base_url: str = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000")).rstrip("/")

        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            available = [m["id"] for m in resp.json().get("data", [])]
            if model not in available:
                logger.warning("'%s' not on server. Available: %s", model, available)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to vLLM at {self.base_url}")
        except Exception as e:
            logger.warning("Could not verify vLLM server: %s", e)
        logger.info("vLLM backend initialized with model '%s'", model)

    def generate(self, prompts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        max_tokens = int(config.get("max_tokens", config.get("max_output_tokens", 512)))
        temperature = float(config.get("temperature", 0.0))
        results = []
        for prompt in prompts:
            try:
                results.append(self._generate_single(prompt, max_tokens, temperature))
            except Exception as e:
                logger.error("vLLM generation failed: %s", e)
                results.append({"text": "", "latency_ms": 0.0, "extra": {"error": repr(e)}})
        return results

    def _generate_single(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
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

        with requests.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=300,
        ) as resp:
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
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if choices and t_first is None and choices[0].get("text"):
                    t_first = time.perf_counter()
                for choice in choices:
                    t = choice.get("text", "")
                    if t:
                        chunks.append(t)
                if "usage" in chunk:
                    usage_data = chunk["usage"]

        t1 = time.perf_counter()
        text = "".join(chunks)
        total_ms = (t1 - t0) * 1000.0

        result: Dict[str, Any] = {
            "text": text,
            "latency_ms": total_ms,
            "extra": {},
        }

        # only report TTFT if we actually measured first-token arrival
        if t_first is not None:
            result["ttft_ms"] = (t_first - t0) * 1000.0
            result["generation_ms"] = (t1 - t_first) * 1000.0

        # only report token counts if the server returned them
        if usage_data:
            result["extra"]["usage"] = {
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return result
