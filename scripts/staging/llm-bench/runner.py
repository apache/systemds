import argparse
import hashlib
import importlib
import json
import logging
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import yaml

from evaluation.perf import perf_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"name"}
VALID_WORKLOADS = {"math", "summarization", "reasoning", "json_extraction"}
VALID_BACKENDS = {"openai", "ollama", "vllm", "mlx"}


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate workload config against expected schema."""
    missing = REQUIRED_CONFIG_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    name = cfg.get("name", "")
    if name not in VALID_WORKLOADS:
        raise ValueError(f"Unknown workload '{name}'. Valid: {VALID_WORKLOADS}")
    dataset_cfg = cfg.get("dataset", {})
    n = dataset_cfg.get("n_samples")
    if n is not None and (not isinstance(n, int) or n < 1):
        raise ValueError(f"n_samples must be a positive integer, got: {n}")


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def create_backend(backend_name: str, model: str, cfg: Dict[str, Any]):
    """Factory function to create the appropriate backend instance."""
    if backend_name not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend '{backend_name}'. Valid: {VALID_BACKENDS}")

    if backend_name == "openai":
        from backends.openai_backend import OpenAIBackend
        backend = OpenAIBackend()
        backend_cfg = cfg.get("openai", {})
        if model:
            backend_cfg = {**backend_cfg, "model": model}
        backend_model = backend_cfg.get("model", "unknown")
        return backend, backend_cfg, backend_model

    # All local backends require --model
    if not model:
        raise RuntimeError(f"--model is required for {backend_name} backend.")

    if backend_name == "mlx":
        from backends.mlx_backend import MLXBackend
        backend = MLXBackend(model)
    elif backend_name == "ollama":
        from backends.ollama_backend import OllamaBackend
        backend = OllamaBackend(model)
    elif backend_name == "vllm":
        from backends.vllm_backend import VLLMBackend
        backend = VLLMBackend(model)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    backend_cfg = cfg.get("generation", {})
    return backend, backend_cfg, model


# ---------------------------------------------------------------------------
# GPU profiling
# ---------------------------------------------------------------------------

def gpu_stats() -> Optional[Dict[str, Any]]:
    """Collect GPU stats via pynvml if available."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpus.append({
                "index": i,
                "name": name,
                "memory_total_mb": mem_info.total / 1024 / 1024,
                "memory_used_mb": mem_info.used / 1024 / 1024,
                "memory_free_mb": mem_info.free / 1024 / 1024,
                "gpu_utilization_pct": util.gpu,
                "memory_utilization_pct": util.memory,
            })
        pynvml.nvmlShutdown()
        return {"gpu_count": count, "gpus": gpus}
    except ImportError:
        logger.debug("pynvml not installed, skipping GPU profiling")
        return None
    except Exception as e:
        logger.debug("GPU profiling failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------

class ResourceMonitor:

    def __init__(self):
        self.process = psutil.Process()
        self.running = False
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.initial_memory = 0.0

    def start(self):
        self.running = True
        self.memory_samples = []
        self.cpu_samples = []
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024

        def _poll():
            while self.running:
                try:
                    self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)
                    self.cpu_samples.append(self.process.cpu_percent())
                except Exception:
                    pass
                time.sleep(0.5)

        self.thread = threading.Thread(target=_poll, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1)
        return {
            "memory_mb_initial": self.initial_memory,
            "memory_mb_peak": max(self.memory_samples) if self.memory_samples else 0,
            "memory_mb_avg": sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            "cpu_percent_avg": sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def json_safe(x):
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    if hasattr(x, "model_dump"):
        return json_safe(x.model_dump())
    if hasattr(x, "dict"):
        return json_safe(x.dict())
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_manifest(out_dir: Path, workload_path: Path, backend: str, model: str) -> None:
    git_hash = None
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        git_hash = r.stdout.strip()
    except Exception:
        pass

    manifest = {
        "git_commit_hash": git_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": {"os": platform.system(), "architecture": platform.machine()},
        "backend": backend,
        "model": model,
        "workload_config_path": str(workload_path.resolve()),
        "workload_config_sha256": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
    }

    # GPU info
    gpu_info = gpu_stats()
    if gpu_info:
        manifest["gpu"] = gpu_info

    write_json(out_dir / "manifest.json", manifest)


def _aggregate_tokens(outputs):
    """Sum real token counts across outputs. Returns (total_in, total_out) or (None, None)."""
    total_in = 0
    total_out = 0
    any_usage = False
    for o in outputs:
        usage = o.get("extra", {}).get("usage")
        if usage:
            any_usage = True
            total_in += usage.get("input_tokens", 0)
            total_out += usage.get("output_tokens", 0)
    if not any_usage:
        return None, None
    return total_in, total_out


# ---------------------------------------------------------------------------
# Concurrent generation
# ---------------------------------------------------------------------------

def _generate_single(backend, prompt: str, backend_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a single prompt -- used by the concurrent executor."""
    results = backend.generate([prompt], backend_cfg)
    return results[0] if results else {"text": "", "latency_ms": 0.0, "extra": {"error": "empty result"}}


def generate_concurrent(backend, prompts: List[str], backend_cfg: Dict[str, Any],
                        concurrency: int) -> List[Dict[str, Any]]:
    """Run prompts concurrently with up to ``concurrency`` threads."""
    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(_generate_single, backend, p, backend_cfg): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error("Concurrent generation failed for prompt %d: %s", idx, e)
                results[idx] = {"text": "", "latency_ms": 0.0, "extra": {"error": repr(e)}}

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="llm-bench runner")
    parser.add_argument("--backend", required=True, choices=sorted(VALID_BACKENDS))
    parser.add_argument("--workload", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-hour-cost", type=float, default=0.0,
                        help="$/GPU-hour for compute cost estimation (e.g. 2.50 for H100)")
    parser.add_argument("--gpu-count", type=int, default=1,
                        help="Number of GPUs used (for compute cost calculation)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent requests (default: 1 = sequential)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = yaml.safe_load(Path(args.workload).read_text(encoding="utf-8"))
    validate_config(cfg)

    workload_name = cfg["name"]
    try:
        loader_module = importlib.import_module(f"workloads.{workload_name}.loader")
        prompt_module = importlib.import_module(f"workloads.{workload_name}.prompt")
        load_samples = loader_module.load_samples
        make_prompt = prompt_module.make_prompt
    except ImportError as e:
        raise RuntimeError(f"Could not load workload '{workload_name}': {e}")

    backend, backend_cfg, backend_model = create_backend(args.backend, args.model, cfg)

    samples = load_samples(cfg)
    prompts = [make_prompt(s, cfg) for s in samples]
    logger.info("Loaded %d samples for workload '%s'", len(samples), workload_name)

    monitor = ResourceMonitor()
    monitor.start()

    # Snapshot GPU before
    gpu_before = gpu_stats()

    t0 = time.perf_counter()
    try:
        if args.concurrency > 1:
            logger.info("Running %d prompts with concurrency=%d", len(prompts), args.concurrency)
            outputs = generate_concurrent(backend, prompts, backend_cfg, args.concurrency)
        else:
            outputs = backend.generate(prompts, backend_cfg)
    except Exception as e:
        logger.error("Generation failed: %s", e)
        outputs = [{"text": "", "latency_ms": 0.0, "extra": {"error": repr(e)}} for _ in prompts]
    t1 = time.perf_counter()
    wall_s = t1 - t0

    resource_stats = monitor.stop()

    # Snapshot GPU after
    gpu_after = gpu_stats()

    accuracy_check_fn = getattr(loader_module, "accuracy_check", None)

    latencies = []
    check_results = []
    rouge_scores_all = []

    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for s, o in zip(samples, outputs):
            lat = float(o.get("latency_ms", 0.0))
            latencies.append(lat)

            pred = o.get("text", "")
            ref = getattr(s, "reference", "")

            is_correct = None
            if accuracy_check_fn is not None and ref:
                is_correct = accuracy_check_fn(pred, ref)
                check_results.append(is_correct)

                # Capture ROUGE scores for summarization
                rouge = getattr(accuracy_check_fn, "last_rouge_scores", None)
                if rouge:
                    rouge_scores_all.append(dict(rouge))

            extra_data = o.get("extra", {})
            ttft_ms = o.get("ttft_ms") or extra_data.get("ttft_ms")
            gen_ms = o.get("generation_ms") or extra_data.get("generation_ms")

            rec: Dict[str, Any] = {
                "id": s.sid,
                "prediction": pred,
                "reference": ref,
                "latency_ms": lat,
                "extra": json_safe(extra_data),
            }
            if is_correct is not None:
                rec["correct"] = is_correct
            if ttft_ms is not None:
                rec["ttft_ms"] = float(ttft_ms)
            if gen_ms is not None:
                rec["generation_ms"] = float(gen_ms)
            if rouge_scores_all and rouge:
                rec["rouge"] = rouge_scores_all[-1]

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = perf_metrics(latencies, total_wall_s=wall_s)

    # accuracy
    if accuracy_check_fn is not None and check_results:
        correct = sum(1 for c in check_results if c)
        total = len(check_results)
        metrics["accuracy_mean"] = correct / total if total > 0 else 0.0
        metrics["accuracy_count"] = f"{correct}/{total}"

    # ROUGE averages for summarization
    if rouge_scores_all:
        for key in rouge_scores_all[0]:
            vals = [s[key] for s in rouge_scores_all if key in s]
            if vals:
                metrics[f"avg_{key}"] = sum(vals) / len(vals)

    # token totals
    total_in, total_out = _aggregate_tokens(outputs)
    if total_in is not None:
        metrics["total_input_tokens"] = total_in
        metrics["total_output_tokens"] = total_out
        metrics["total_tokens"] = total_in + total_out

    # API cost (OpenAI)
    api_cost = sum(o.get("extra", {}).get("cost_usd", 0.0) for o in outputs)
    if api_cost > 0:
        metrics["api_cost_usd"] = api_cost

    # compute cost (local backends -- user supplies $/GPU-hour)
    if args.gpu_hour_cost > 0:
        gpu_hours = (wall_s / 3600.0) * args.gpu_count
        metrics["gpu_hours"] = gpu_hours
        metrics["compute_cost_usd"] = gpu_hours * args.gpu_hour_cost

    # concurrency info
    if args.concurrency > 1:
        metrics["concurrency"] = args.concurrency

    metrics.update(resource_stats)

    # GPU profiling
    if gpu_before:
        metrics["gpu_info"] = gpu_before
    if gpu_after:
        metrics["gpu_after"] = gpu_after

    write_json(out_dir / "metrics.json", metrics)

    write_json(out_dir / "run_config.json", {
        "backend": args.backend,
        "backend_model": backend_model,
        "workload": cfg.get("name", "unknown"),
        "concurrency": args.concurrency,
    })

    write_manifest(out_dir, Path(args.workload), args.backend, backend_model)

    logger.info("OK: wrote %s", out_dir)
    print(f"OK: wrote {out_dir}")


if __name__ == "__main__":
    main()
