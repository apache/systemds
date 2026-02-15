import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import psutil
import yaml

from evaluation.perf import perf_metrics


class ResourceMonitor:

    def __init__(self):
        self.process = psutil.Process()
        self.running = False
        self.memory_samples = []
        self.cpu_samples = []
        self.initial_memory = 0.0

    def start(self):
        self.running = True
        self.memory_samples = []
        self.cpu_samples = []
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024

        def _poll():
            while self.running:
                self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)
                self.cpu_samples.append(self.process.cpu_percent())
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


def main():
    parser = argparse.ArgumentParser(description="llm-bench runner")
    parser.add_argument("--backend", required=True, choices=["openai", "ollama", "vllm", "mlx"])
    parser.add_argument("--workload", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-hour-cost", type=float, default=0.0,
                        help="$/GPU-hour for compute cost estimation (e.g. 2.50 for H100)")
    parser.add_argument("--gpu-count", type=int, default=1,
                        help="Number of GPUs used (for compute cost calculation)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = yaml.safe_load(Path(args.workload).read_text(encoding="utf-8"))

    workload_name = cfg.get("name", "summarization")
    try:
        loader_module = importlib.import_module(f"workloads.{workload_name}.loader")
        prompt_module = importlib.import_module(f"workloads.{workload_name}.prompt")
        load_samples = loader_module.load_samples
        make_prompt = prompt_module.make_prompt
    except ImportError as e:
        raise RuntimeError(f"Could not load workload '{workload_name}': {e}")

    if args.backend == "mlx":
        if not args.model:
            raise RuntimeError("--model is required for mlx backend.")
        from backends.mlx_backend import MLXBackend
        backend = MLXBackend(args.model)
        backend_cfg = cfg.get("generation", {})
        backend_model = args.model
    elif args.backend == "ollama":
        if not args.model:
            raise RuntimeError("--model is required for ollama backend.")
        from backends.ollama_backend import OllamaBackend
        backend = OllamaBackend(args.model)
        backend_cfg = cfg.get("generation", {})
        backend_model = args.model
    elif args.backend == "vllm":
        if not args.model:
            raise RuntimeError("--model is required for vllm backend.")
        from backends.vllm_backend import VLLMBackend
        backend = VLLMBackend(args.model)
        backend_cfg = cfg.get("generation", {})
        backend_model = args.model
    else:
        from backends.openai_backend import OpenAIBackend
        backend = OpenAIBackend()
        backend_cfg = cfg.get("openai", {})
        if args.model:
            backend_cfg = {**backend_cfg, "model": args.model}
        backend_model = backend_cfg.get("model", "unknown")

    samples = load_samples(cfg)
    prompts = [make_prompt(s, cfg) for s in samples]

    monitor = ResourceMonitor()
    monitor.start()

    t0 = time.perf_counter()
    try:
        outputs = backend.generate(prompts, backend_cfg)
    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        outputs = [{"text": "", "latency_ms": 0.0, "extra": {"error": repr(e)}} for _ in prompts]
    t1 = time.perf_counter()
    wall_s = t1 - t0

    resource_stats = monitor.stop()

    accuracy_check_fn = getattr(loader_module, "accuracy_check", None)

    latencies = []
    check_results = []

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

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = perf_metrics(latencies, total_wall_s=wall_s)

    # accuracy (or quality-gate for summarization)
    if accuracy_check_fn is not None and check_results:
        correct = sum(1 for c in check_results if c)
        total = len(check_results)
        metrics["accuracy_mean"] = correct / total if total > 0 else 0.0
        metrics["accuracy_count"] = f"{correct}/{total}"

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

    metrics.update(resource_stats)

    write_json(out_dir / "metrics.json", metrics)

    write_json(out_dir / "run_config.json", {
        "backend": args.backend,
        "backend_model": backend_model,
        "workload": cfg.get("name", "unknown"),
    })

    write_manifest(out_dir, Path(args.workload), args.backend, backend_model)

    print(f"OK: wrote {out_dir}")


if __name__ == "__main__":
    main()
