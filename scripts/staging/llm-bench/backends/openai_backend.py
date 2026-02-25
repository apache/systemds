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

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


# Pricing per million tokens (USD).
# Reference: https://openai.com/api/pricing/
# Last verified: 2026-02-18. OpenAI does not expose a pricing API, so this
# table must be updated manually when prices change.
# To add a missing model without editing this file, create a file called
# pricing.json next to this file with the format:
#   {"my-model": {"input": 1.00, "output": 2.00}}
# It will be merged with the table below at import time.
PRICING_LAST_UPDATED = "2026-02-18"
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4.1-mini":            {"input": 0.40, "output": 1.60},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60},
    "gpt-4.1":                 {"input": 2.00, "output": 8.00},
    "gpt-4.1-2025-04-14":      {"input": 2.00, "output": 8.00},
    "gpt-4.1-nano":            {"input": 0.10, "output": 0.40},
    "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
    "gpt-4o":                  {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":             {"input": 0.15, "output": 0.60},
}

_pricing_override = Path(__file__).parent / "pricing.json"
if _pricing_override.exists():
    try:
        _extra = json.loads(_pricing_override.read_text(encoding="utf-8"))
        PRICING.update(_extra)
        logger.debug("Loaded %d pricing overrides from %s", len(_extra), _pricing_override)
    except Exception as _e:
        logger.warning("Could not load %s: %s", _pricing_override, _e)


class OpenAIBackend:

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        model = config.get("model", "gpt-4.1-mini")
        max_output_tokens = int(config.get("max_output_tokens", config.get("max_tokens", 256)))
        temperature = config.get("temperature", 0.0)
        use_streaming = config.get("streaming", False)
        max_retries = int(config.get("max_retries", 5))
        base_sleep = float(config.get("base_sleep_s", 0.5))

        results = []

        for prompt in prompts:
            last_err = None
            for attempt in range(max_retries):
                try:
                    if use_streaming:
                        result = self._generate_streaming(
                            prompt, model, max_output_tokens, temperature
                        )
                    else:
                        result = self._generate_non_streaming(
                            prompt, model, max_output_tokens, temperature
                        )
                    
                    results.append(result)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(base_sleep * (2**attempt))

            if last_err is not None:
                results.append(
                    {
                        "text": "",
                        "latency_ms": 0.0,
                        "extra": {"error": repr(last_err)},
                    }
                )

        return results
    
    def _generate_non_streaming(self, prompt: str, model: str, max_output_tokens: int, temperature: float) -> Dict[str, Any]:
        t0 = time.perf_counter()
        resp = self.client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        t1 = time.perf_counter()

        text = ""
        try:
            text = resp.output_text
        except Exception:
            text = str(resp)

        extra: Dict[str, Any] = {}
        usage = getattr(resp, "usage", None)
        if usage is not None:
            usage_data = self._extract_usage(usage)
            if usage_data is not None:
                extra["usage"] = usage_data
                cost = self._calculate_cost(usage_data, model)
                if cost is not None:
                    extra["cost_usd"] = cost
        extra["response_id"] = getattr(resp, "id", None)

        return {
            "text": text,
            "latency_ms": (t1 - t0) * 1000.0,
            "extra": extra,
        }
    
    def _generate_streaming(self, prompt: str, model: str, max_output_tokens: int, temperature: float) -> Dict[str, Any]:
        t0 = time.perf_counter()
        stream = self.client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stream=True,
        )
        
        t_first = None
        t_final = None
        full_text = ""
        response_id = None
        usage_data = None
        
        for event in stream:
            if event.type == "response.output_text.delta":
                if t_first is None:
                    t_first = time.perf_counter()
                full_text += event.delta
            
            elif event.type == "response.completed":
                t_final = time.perf_counter()
                response = getattr(event, "response", None)
                if response is not None:
                    response_id = getattr(response, "id", None)
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        usage_data = self._extract_usage(usage)
                else:
                    response_id = getattr(event, "response_id", None) or getattr(event, "id", None)
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        usage_data = self._extract_usage(usage)
        
        if usage_data is None:
            stream_usage = getattr(stream, "usage", None)
            if stream_usage is not None:
                usage_data = self._extract_usage(stream_usage)
        
        if t_first is None:
            t_first = time.perf_counter()
        if t_final is None:
            t_final = time.perf_counter()
        
        ttft_ms = (t_first - t0) * 1000.0
        generation_ms = (t_final - t_first) * 1000.0
        total_latency_ms = (t_final - t0) * 1000.0
        
        extra: Dict[str, Any] = {
            "ttft_ms": ttft_ms,
            "generation_ms": generation_ms,
            "response_id": response_id,
        }

        if usage_data is not None:
            extra["usage"] = usage_data
            cost = self._calculate_cost(usage_data, model)
            if cost is not None:
                extra["cost_usd"] = cost
        
        return {
            "text": full_text,
            "latency_ms": total_latency_ms, 
            "extra": extra,
        }
    
    def _extract_usage(self, usage: Any) -> Optional[Dict[str, Any]]:
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        elif hasattr(usage, "dict"):
            return usage.dict()
        elif isinstance(usage, dict):
            return usage
        else:
            return {"raw": str(usage)}
    
    def _calculate_cost(self, usage_data: Optional[Dict[str, Any]], model: str) -> Optional[float]:
        if usage_data is None:
            return None
        
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        
        if input_tokens == 0 and output_tokens == 0:
            return None

        prices = PRICING.get(model)
        if prices is None:
            logger.warning(
                "No pricing data for model '%s' (table last updated %s). "
                "Cost will not be reported. Check https://openai.com/api/pricing/ "
                "and update PRICING in openai_backend.py if needed.",
                model, PRICING_LAST_UPDATED,
            )
            return None

        cost = (
            input_tokens * prices["input"] / 1_000_000 +
            output_tokens * prices["output"] / 1_000_000
        )
        return cost