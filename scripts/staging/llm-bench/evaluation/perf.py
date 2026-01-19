from typing import Dict, List
import numpy as np


def perf_metrics(latencies_ms: List[float], total_wall_s: float) -> Dict[str, float]:
    arr = np.array(latencies_ms, dtype=float)
    if len(arr) == 0:
        return {
            "n": 0.0,
            "latency_ms_mean": 0.0,
            "latency_ms_std": 0.0,
            "latency_ms_min": 0.0,
            "latency_ms_max": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
            "latency_ms_cv": 0.0,
            "throughput_req_per_s": 0.0,
        }

    mean = float(arr.mean())
    std = float(arr.std())

    return {
        "n": float(len(arr)),
        "latency_ms_mean": mean,
        "latency_ms_std": std,
        "latency_ms_min": float(arr.min()),
        "latency_ms_max": float(arr.max()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_cv": std / mean if mean > 0 else 0.0,
        "throughput_req_per_s": float(len(arr) / total_wall_s) if total_wall_s > 0 else 0.0,
    }
