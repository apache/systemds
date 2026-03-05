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

"""SystemDS JMLC backend using the native llmPredict built-in."""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Default paths relative to the SystemDS project root
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # llm-bench -> staging -> scripts -> systemds
_DEFAULT_SYSTEMDS_JAR = _PROJECT_ROOT / "target" / "SystemDS.jar"
_DEFAULT_LIB_DIR = _PROJECT_ROOT / "target" / "lib"

# DML script that uses the native llmPredict built-in
_DML_SCRIPT = (
    'prompts = read("prompts", data_type="frame")\n'
    'results = llmPredict(target=prompts, url=$url, model=$model, max_tokens=$mt,'
    ' temperature=$temp, top_p=$tp, concurrency=$conc)\n'
    'write(results, "results")'
)


def _build_classpath(systemds_jar: str, lib_dir: str) -> str:
    jars = [systemds_jar]
    lib_path = Path(lib_dir)
    if lib_path.is_dir():
        jars.extend(str(p) for p in sorted(lib_path.glob("*.jar")))
    return os.pathsep.join(jars)


class SystemDSBackend:

    def __init__(self, model: str):
        self.model = model

        self.systemds_jar = os.environ.get("SYSTEMDS_JAR", str(_DEFAULT_SYSTEMDS_JAR))
        self.lib_dir = os.environ.get("SYSTEMDS_LIB", str(_DEFAULT_LIB_DIR))
        self.inference_url = os.environ.get(
            "LLM_INFERENCE_URL", "http://localhost:8080/v1/completions")

        if not Path(self.systemds_jar).exists():
            raise RuntimeError(
                f"SystemDS JAR not found at {self.systemds_jar}. "
                "Build with: mvn package -DskipTests  "
                "Or set SYSTEMDS_JAR env var."
            )

        classpath = _build_classpath(self.systemds_jar, self.lib_dir)
        logger.info("Starting JVM with classpath: %s ... (%d JARs)",
                     self.systemds_jar, classpath.count(os.pathsep) + 1)

        from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway

        self._gw_port = launch_gateway(
            classpath=classpath,
            die_on_exit=True,
            javaopts=["--add-modules=jdk.incubator.vector"],
            redirect_stdout=subprocess.sys.stdout,
            redirect_stderr=subprocess.sys.stderr,
        )
        self._gateway = JavaGateway(
            gateway_parameters=GatewayParameters(port=self._gw_port)
        )

        self._jvm = self._gateway.jvm
        self._connection = self._jvm.org.apache.sysds.api.jmlc.Connection()
        # cache compiled scripts -- $-args are compile-time, so recompile only when params change
        self._script_cache: dict = {}

        logger.info("SystemDS JMLC backend initialized (model=%s, url=%s)",
                     model, self.inference_url)

    def generate(self, prompts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        max_tokens = int(config.get("max_tokens", config.get("max_output_tokens", 512)))
        temperature = float(config.get("temperature", 0.0))
        top_p = float(config.get("top_p", 0.9))
        concurrency = int(config.get("concurrency",
            os.environ.get("SYSTEMDS_CONCURRENCY", "1")))

        jvm = self._jvm

        t_pipeline_start = time.perf_counter()

        # --- Phase 1: DML compilation (or cache hit) ---
        t_compile_start = time.perf_counter()
        script_key = (self.inference_url, self.model, max_tokens, temperature, top_p, concurrency)
        cache_hit = script_key in self._script_cache
        if cache_hit:
            ps = self._script_cache[script_key]
            logger.debug("Reusing cached PreparedScript for key %s", script_key)
        else:
            args = self._gateway.jvm.java.util.HashMap()
            args.put("$url", self.inference_url)
            args.put("$model", self.model)
            args.put("$mt", str(max_tokens))
            args.put("$temp", str(temperature))
            args.put("$tp", str(top_p))
            args.put("$conc", str(concurrency))

            inputs = self._gateway.new_array(jvm.java.lang.String, 1)
            inputs[0] = "prompts"
            outputs = self._gateway.new_array(jvm.java.lang.String, 1)
            outputs[0] = "results"

            ps = self._connection.prepareScript(_DML_SCRIPT, args, inputs, outputs)
            self._script_cache[script_key] = ps
            logger.debug("Compiled and cached new PreparedScript for key %s", script_key)
        t_compile_end = time.perf_counter()

        # --- Phase 2: Py4J marshalling (prompts -> Java) ---
        t_marshal_start = time.perf_counter()
        n = len(prompts)
        prompt_data = self._gateway.new_array(jvm.java.lang.String, n, 1)
        for i, p in enumerate(prompts):
            prompt_data[i][0] = p
        ps.setFrame("prompts", prompt_data)
        t_marshal_end = time.perf_counter()

        # --- Phase 3: Java execution (DML -> llmPredict -> HTTP) ---
        t_exec_start = time.perf_counter()
        try:
            rv = ps.executeScript()
        except Exception as e:
            err_msg = str(e)
            # unwrap Py4J-wrapped Java exceptions
            if "java.net.ConnectException" in err_msg:
                raise RuntimeError(
                    f"Inference server unreachable at {self.inference_url}. "
                    "Is the LLM server running?"
                ) from e
            if "java.net.SocketTimeoutException" in err_msg:
                raise RuntimeError(
                    "Inference server timed out. The server may be overloaded "
                    "or the read timeout (300 s) was exceeded."
                ) from e
            raise RuntimeError(
                f"SystemDS executeScript failed: {err_msg}"
            ) from e
        t_exec_end = time.perf_counter()

        # --- Phase 4: Py4J unmarshalling (Java FrameBlock -> Python) ---
        t_unmarshal_start = time.perf_counter()
        frame_block = rv.getFrameBlock("results")
        t_unmarshal_end = time.perf_counter()

        t_pipeline_end = time.perf_counter()

        compile_ms = (t_compile_end - t_compile_start) * 1000.0
        marshal_ms = (t_marshal_end - t_marshal_start) * 1000.0
        exec_wall_ms = (t_exec_end - t_exec_start) * 1000.0
        unmarshal_ms = (t_unmarshal_end - t_unmarshal_start) * 1000.0
        pipeline_wall_ms = (t_pipeline_end - t_pipeline_start) * 1000.0

        raw = []
        for i in range(n):
            text = str(frame_block.get(i, 1))
            try:
                java_http_ms = int(float(str(frame_block.get(i, 2))))
            except (ValueError, TypeError):
                java_http_ms = 0
            try:
                input_tokens = int(float(str(frame_block.get(i, 3))))
            except (ValueError, TypeError):
                input_tokens = 0
            try:
                output_tokens = int(float(str(frame_block.get(i, 4))))
            except (ValueError, TypeError):
                output_tokens = 0
            raw.append((text, float(java_http_ms), input_tokens, output_tokens))

        # per-prompt latency = java_http_ms + share of pipeline overhead
        # with concurrency > 1, HTTP calls overlap so just use pipeline_wall_ms / n
        total_java_http = sum(r[1] for r in raw)
        overhead_ms = pipeline_wall_ms - total_java_http
        use_per_prompt = concurrency <= 1 and overhead_ms >= 0
        if not use_per_prompt:
            logger.warning(
                "Per-prompt latency uses amortised pipeline_wall_ms/n "
                "(concurrency=%d, overhead=%.1fms). Individual HTTP times "
                "overlap and cannot be attributed per-prompt.",
                concurrency, overhead_ms,
            )

        results = []
        for text, java_http_ms, input_tokens, output_tokens in raw:
            if use_per_prompt:
                lat = java_http_ms + overhead_ms / n
            else:
                lat = pipeline_wall_ms / n
            results.append({
                "text": text,
                "latency_ms": lat,
                "extra": {
                    "java_http_ms": java_http_ms,
                    "compile_ms": compile_ms,
                    "compile_cache_hit": cache_hit,
                    "marshal_ms": marshal_ms,
                    "unmarshal_ms": unmarshal_ms,
                    "exec_wall_ms": exec_wall_ms / n,
                    "pipeline_wall_ms": pipeline_wall_ms,
                    "pipeline_overhead_ms": max(0.0, overhead_ms),
                    "concurrency": concurrency,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                },
            })

        avg_java_http_ms = sum(r["extra"]["java_http_ms"] for r in results) / n
        logger.info(
            "llmPredict: %d prompts | pipeline=%.1fms | "
            "compile=%.1fms (%s) | marshal=%.1fms | exec=%.1fms | "
            "unmarshal=%.1fms | java_http=%.1fms/prompt (avg)",
            n, pipeline_wall_ms,
            compile_ms, "hit" if cache_hit else "miss",
            marshal_ms, exec_wall_ms,
            unmarshal_ms, avg_java_http_ms,
        )
        return results

    def close(self):
        try:
            if hasattr(self, "_connection") and self._connection is not None:
                self._connection.close()
        except Exception as e:
            logger.debug("Error closing JMLC connection: %s", e)
        try:
            if hasattr(self, "_gateway") and self._gateway is not None:
                self._gateway.shutdown()
        except Exception as e:
            logger.debug("Error shutting down gateway: %s", e)

    def __del__(self):
        self.close()
