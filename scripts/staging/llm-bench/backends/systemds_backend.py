"""SystemDS JMLC backend for the LLM benchmark runner.

Data flow:
  Python benchmark -> Py4J -> Java JMLC Connection
  -> PreparedScript.generateBatchWithMetrics()
  -> LLMCallback -> Python llm_worker.py (HuggingFace)
  -> FrameBlock [prompt, generated_text, time_ms, input_tokens, output_tokens]
"""

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
_DEFAULT_WORKER_SCRIPT = _PROJECT_ROOT / "src" / "main" / "python" / "llm_worker.py"


def _build_classpath(systemds_jar: str, lib_dir: str) -> str:
    """Build JVM classpath from SystemDS JAR and its dependency directory."""
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
        self.worker_script = os.environ.get("LLM_WORKER_SCRIPT", str(_DEFAULT_WORKER_SCRIPT))

        if not Path(self.systemds_jar).exists():
            raise RuntimeError(
                f"SystemDS JAR not found at {self.systemds_jar}. "
                "Build with: mvn package -DskipTests  "
                "Or set SYSTEMDS_JAR env var."
            )
        if not Path(self.worker_script).exists():
            raise RuntimeError(
                f"LLM worker script not found at {self.worker_script}. "
                "Or set LLM_WORKER_SCRIPT env var."
            )

        classpath = _build_classpath(self.systemds_jar, self.lib_dir)
        logger.info("Starting JVM with classpath: %s ... (%d JARs)",
                     self.systemds_jar, classpath.count(os.pathsep) + 1)

        # Ensure the current virtualenv (if any) is on PATH so that
        # Connection.findPythonCommand() finds the correct python3 with
        # torch/transformers installed.
        import sys
        venv_bin = Path(sys.executable).parent
        current_path = os.environ.get("PATH", "")
        if str(venv_bin) not in current_path:
            os.environ["PATH"] = str(venv_bin) + os.pathsep + current_path

        from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway

        self._gw_port = launch_gateway(
            classpath=classpath,
            die_on_exit=True,
            redirect_stdout=subprocess.sys.stdout,
            redirect_stderr=subprocess.sys.stderr,
        )
        self._gateway = JavaGateway(
            gateway_parameters=GatewayParameters(port=self._gw_port)
        )

        logger.info("Creating JMLC Connection and loading model '%s' ...", model)
        jvm = self._gateway.jvm
        self._jvm = jvm

        self._connection = jvm.org.apache.sysds.api.jmlc.Connection()
        self._llm_callback = self._connection.loadModel(model, self.worker_script)

        # Set up a PreparedScript with the LLM worker attached so we can
        # call generateBatchWithMetrics() -- the FrameBlock-based API.
        dummy_script = "x = 1;"
        self._ps = self._connection.prepareScript(
            dummy_script,
            self._gateway.new_array(jvm.java.lang.String, 0),
            self._gateway.new_array(jvm.java.lang.String, 0),
        )
        self._ps.setLLMWorker(self._llm_callback)

        logger.info("SystemDS JMLC backend initialized (model=%s)", model)

    def generate(self, prompts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        max_tokens = int(config.get("max_tokens", config.get("max_output_tokens", 512)))
        temperature = float(config.get("temperature", 0.0))
        top_p = float(config.get("top_p", 0.9))

        n = len(prompts)
        java_prompts = self._gateway.new_array(self._jvm.java.lang.String, n)
        for i, p in enumerate(prompts):
            java_prompts[i] = p

        t0 = time.perf_counter()
        frame_block = self._ps.generateBatchWithMetrics(
            java_prompts, max_tokens, temperature, top_p
        )
        t1 = time.perf_counter()
        batch_wall_ms = (t1 - t0) * 1000.0

        results = []
        for i in range(n):
            text = str(frame_block.get(i, 1))
            java_time_ms = int(str(frame_block.get(i, 2)))
            input_tokens = int(str(frame_block.get(i, 3)))
            output_tokens = int(str(frame_block.get(i, 4)))

            results.append({
                "text": text,
                "latency_ms": float(java_time_ms),
                "extra": {
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                },
            })

        logger.info(
            "FrameBlock batch: %d prompts in %.1fms (%.1fms/prompt)",
            n, batch_wall_ms, batch_wall_ms / n,
        )
        return results

    def close(self):
        """Shut down the JMLC connection and JVM gateway."""
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
