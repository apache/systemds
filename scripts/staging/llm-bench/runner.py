import argparse
import importlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone

import psutil
import yaml

from evaluation.perf import perf_metrics


class ResourceMonitor:
    """Monitor CPU and memory usage during benchmark execution."""
    
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
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def monitor():
            while self.running:
                self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)
                self.cpu_samples.append(self.process.cpu_percent())
                time.sleep(0.5)
        
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
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
    # pydantic-like objects
    if hasattr(x, "model_dump"):
        return json_safe(x.model_dump())
    if hasattr(x, "dict"):
        return json_safe(x.dict())
    return str(x)

def write_manifest(out_dir: Path, workload_path: Path, backend: str, model: str) -> None:
    git_commit_hash = None
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_commit_hash = r.stdout.strip()
    except Exception:
        git_commit_hash = None

    workload_bytes = workload_path.read_bytes()
    workload_sha256 = hashlib.sha256(workload_bytes).hexdigest()

    manifest = {
        "git_commit_hash": git_commit_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": {
            "os": platform.system(),
            "architecture": platform.machine(),
        },
        "backend": backend,
        "model": model,
        "workload_config_path": str(workload_path.resolve()),
        "workload_config_sha256": workload_sha256,
    }
    write_json(out_dir / "manifest.json", manifest)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="systemds-bench-gpt runner")
    parser.add_argument("--backend", required=True, choices=["openai", "ollama", "vllm", "mlx"],
                        help="Backend: openai (API), ollama (local), vllm (server), mlx (Apple Silicon)")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = yaml.safe_load(Path(args.workload).read_text(encoding="utf-8"))
    
    # dynamically load the workload module based on config name
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
            raise RuntimeError("--model is required for ollama backend (e.g., llama3.2, mistral, phi3)")
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
    else:  # openai
        from backends.openai_backend import OpenAIBackend
        backend = OpenAIBackend()
        backend_cfg = cfg.get("openai", {})
        if args.model:
            backend_cfg = {**backend_cfg, "model": args.model}
        backend_model = backend_cfg.get("model", "unknown")

    samples = load_samples(cfg)
    prompts = [make_prompt(s, cfg) for s in samples]

    # start resource monitoring
    monitor = ResourceMonitor()
    monitor.start()

    t0 = time.perf_counter()
    outputs = backend.generate(prompts, backend_cfg)
    t1 = time.perf_counter()

    # stop monitoring and get resource stats
    resource_stats = monitor.stop()

    # check if workload has accuracy_check function
    accuracy_check_fn = getattr(loader_module, "accuracy_check", None)
    
    latencies = []
    predictions_for_accuracy = []  # store (prediction, reference) pairs for accuracy calc
    
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for s, o in zip(samples, outputs):
            lat = float(o.get("latency_ms", 0.0))
            latencies.append(lat)
            
            prediction_text = o.get("text", "")
            reference_text = getattr(s, "reference", "")
            
            # check accuracy and store for aggregation
            is_correct = None
            if accuracy_check_fn is not None and reference_text:
                is_correct = accuracy_check_fn(prediction_text, reference_text)
                predictions_for_accuracy.append((prediction_text, reference_text))
            
            # extract TTFT metrics (can be at top level or in extra dict)
            extra_data = o.get("extra", {})
            ttft_ms = o.get("ttft_ms") or extra_data.get("ttft_ms")
            generation_ms = o.get("generation_ms") or extra_data.get("generation_ms")
            
            rec = {
                "id": s.sid,
                "prediction": prediction_text,
                "reference": reference_text,
                "latency_ms": lat,
                "extra": json_safe(extra_data),
            }
            
            # add correctness field for per-sample debugging
            if is_correct is not None:
                rec["correct"] = is_correct
            
            # add TTFT metrics at top level if available (easier for aggregate.py/report.py)
            if ttft_ms is not None:
                rec["ttft_ms"] = float(ttft_ms)
            if generation_ms is not None:
                rec["generation_ms"] = float(generation_ms)
            
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = perf_metrics(latencies, total_wall_s=(t1 - t0))
    
    # calculate accuracy if accuracy_check function is available
    if accuracy_check_fn is not None and predictions_for_accuracy:
        correct = sum(1 for pred, ref in predictions_for_accuracy if accuracy_check_fn(pred, ref))
        total = len(predictions_for_accuracy)
        metrics["accuracy_mean"] = correct / total if total > 0 else 0.0
        metrics["accuracy_count"] = f"{correct}/{total}"
    
    # aggregate cost from all outputs
    total_cost = sum(o.get("extra", {}).get("cost_usd", 0.0) for o in outputs)
    total_tokens = sum(o.get("extra", {}).get("usage", {}).get("total_tokens", 0) for o in outputs)
    
    if total_cost > 0:
        metrics["cost_total_usd"] = total_cost
        metrics["cost_per_1m_tokens"] = (total_cost / total_tokens * 1_000_000) if total_tokens > 0 else 0.0
    
    # add resource usage stats
    metrics.update(resource_stats)
    
    write_json(out_dir / "metrics.json", metrics)
    
    # add run_config.json for reporting
    run_config = {
        "backend": args.backend,
        "backend_model": backend_model,
        "workload": cfg.get("name", "unknown"),
    }
    write_json(out_dir / "run_config.json", run_config)
    
    write_manifest(out_dir, Path(args.workload), args.backend, backend_model)
    
    print(f"OK: wrote {out_dir}")


if __name__ == "__main__":
    main()