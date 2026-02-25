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

"""MLX backend -- Apple Silicon local inference via mlx-lm."""

import logging
import time
from typing import Any, Dict, List

from mlx_lm import load, stream_generate

logger = logging.getLogger(__name__)


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
                results.append(self._generate_single(p, max_tokens, temperature))
            except Exception as e:
                logger.error("MLX generation failed: %s", e)
                results.append({"text": "", "latency_ms": 0.0, "extra": {"error": repr(e)}})

        return results

    def _generate_single(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        t0 = time.perf_counter()
        t_first = None
        chunks: List[str] = []

        for token_text in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            temp=temperature,
        ):
            if t_first is None:
                t_first = time.perf_counter()
            chunks.append(token_text)

        t1 = time.perf_counter()
        out = "".join(chunks)
        total_ms = (t1 - t0) * 1000.0

        if t_first is None:
            t_first = t1

        ttft_ms = (t_first - t0) * 1000.0
        gen_ms = (t1 - t_first) * 1000.0

        extra: Dict[str, Any] = {}
        try:
            in_tok = len(self.tokenizer.encode(prompt))
            out_tok = len(chunks)
            extra["usage"] = {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            }
        except Exception:
            pass

        return {
            "text": out,
            "latency_ms": total_ms,
            "ttft_ms": ttft_ms,
            "generation_ms": gen_ms,
            "extra": extra,
        }
