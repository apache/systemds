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
    'X = read("prompts", data_type="frame")\n'
    'R = llmPredict(target=X, url=$url, max_tokens=$mt,'
    ' temperature=$temp, top_p=$tp, concurrency=$conc)\n'
    'write(R, "results")'
)


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
            redirect_stdout=subprocess.sys.stdout,
            redirect_stderr=subprocess.sys.stderr,
        )
        self._gateway = JavaGateway(
            gateway_parameters=GatewayParameters(port=self._gw_port)
        )

        self._jvm = self._gateway.jvm
        self._connection = self._jvm.org.apache.sysds.api.jmlc.Connection()

        logger.info("SystemDS JMLC backend initialized (model=%s, url=%s)",
                     model, self.inference_url)

    def generate(self, prompts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        max_tokens = int(config.get("max_tokens", config.get("max_output_tokens", 512)))
        temperature = float(config.get("temperature", 0.0))
        top_p = float(config.get("top_p", 0.9))
        concurrency = int(config.get("concurrency",
            os.environ.get("SYSTEMDS_CONCURRENCY", "1")))

        jvm = self._jvm

        args = self._gateway.jvm.java.util.HashMap()
        args.put("$url", self.inference_url)
        args.put("$mt", str(max_tokens))
        args.put("$temp", str(temperature))
        args.put("$tp", str(top_p))
        args.put("$conc", str(concurrency))

        # Prepare DML script with llmPredict built-in
        inputs = self._gateway.new_array(jvm.java.lang.String, 1)
        inputs[0] = "prompts"
        outputs = self._gateway.new_array(jvm.java.lang.String, 1)
        outputs[0] = "results"

        ps = self._connection.prepareScript(_DML_SCRIPT, args, inputs, outputs)

        # Build prompt frame (n x 1 String[][])
        n = len(prompts)
        prompt_data = self._gateway.new_array(jvm.java.lang.String, n, 1)
        for i, p in enumerate(prompts):
            prompt_data[i][0] = p
        ps.setFrame("prompts", prompt_data)

        # Execute through full SystemDS pipeline
        t0 = time.perf_counter()
        rv = ps.executeScript()
        t1 = time.perf_counter()
        batch_wall_ms = (t1 - t0) * 1000.0

        frame_block = rv.getFrameBlock("results")

        results = []
        for i in range(n):
            text = str(frame_block.get(i, 1))
            per_prompt_ms = int(str(frame_block.get(i, 2)))
            input_tokens = int(str(frame_block.get(i, 3)))
            output_tokens = int(str(frame_block.get(i, 4)))

            results.append({
                "text": text,
                "latency_ms": float(per_prompt_ms),
                "extra": {
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                },
            })

        logger.info(
            "llmPredict: %d prompts in %.1fms (%.1fms/prompt)",
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
