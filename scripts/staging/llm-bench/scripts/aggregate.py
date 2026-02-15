
import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple

# allow running from project root (python scripts/aggregate.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import read_json, iter_run_dirs, manifest_timestamp, token_stats, ttft_stats

def sort_key(run_dir: Path) -> Tuple[int, str, str]:
    """
    Sort runs chronologically by manifest timestamp if available.
    Missing timestamp => later in ordering and sorted by name.
    """
    ts = manifest_timestamp(run_dir)
    missing = 1 if ts == "" else 0
    return (missing, ts, run_dir.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate benchmark runs under results/ into CSV.")
    parser.add_argument("--results-dir", default="results", help="Directory containing run folders (default: results)")
    parser.add_argument("--out", default="-", help="Output CSV path or '-' for stdout (default: '-')")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_dirs = list(iter_run_dirs(results_dir))
    run_dirs.sort(key=sort_key)

    if not run_dirs:
        print(f"Error: no valid run directories found under {results_dir}/", file=sys.stderr)
        return 1

    header = [
        "run_dir",
        "ts",
        "backend",
        "backend_model",
        "workload",
        "n",
        "accuracy_mean",
        "accuracy_count",
        "cost_total_usd",
        "cost_per_1m_tokens",
        "memory_mb_peak",
        "cpu_percent_avg",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_ms_min",
        "latency_ms_max",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_cv",
        "throughput_req_per_s",
        "total_tokens",
        "avg_tokens",
        "total_input_tokens",
        "total_output_tokens",
        "ttft_ms_mean",
        "generation_ms_mean",
        "concurrency",
        "rouge1_f",
        "rouge2_f",
        "rougeL_f",
    ]

    if args.out == "-":
        out_f = sys.stdout
        close_after = False
    else:
        out_f = open(args.out, "w", encoding="utf-8", newline="")
        close_after = True

    try:
        writer = csv.writer(out_f)
        writer.writerow(header)

        for run_dir in run_dirs:
            try:
                metrics = read_json(run_dir / "metrics.json")
                cfg = read_json(run_dir / "run_config.json")
                ts = manifest_timestamp(run_dir)
                total, avg, total_in, total_out = token_stats(run_dir / "samples.jsonl")
                ttft_mean, gen_mean = ttft_stats(run_dir / "samples.jsonl")

                # get accuracy from metrics.json (stored by runner)
                accuracy_mean = metrics.get("accuracy_mean")
                accuracy_count = metrics.get("accuracy_count", "")
                
                # get cost from metrics.json
                cost_total = metrics.get("cost_total_usd")
                cost_per_1m = metrics.get("cost_per_1m_tokens")
                
                # get resource usage metrics
                memory_mb_peak = metrics.get("memory_mb_peak")
                cpu_percent_avg = metrics.get("cpu_percent_avg")
                
                # get latency variance metrics
                lat_std = metrics.get("latency_ms_std")
                lat_min = metrics.get("latency_ms_min")
                lat_max = metrics.get("latency_ms_max")
                lat_cv = metrics.get("latency_ms_cv")
                
                row = [
                    run_dir.name,
                    ts,
                    cfg.get("backend", ""),
                    cfg.get("backend_model", ""),
                    cfg.get("workload", ""),
                    metrics.get("n", ""),
                    "" if accuracy_mean is None else f"{accuracy_mean:.4f}",
                    accuracy_count,
                    "" if cost_total is None else f"{cost_total:.6f}",
                    "" if cost_per_1m is None else f"{cost_per_1m:.4f}",
                    "" if memory_mb_peak is None else f"{memory_mb_peak:.1f}",
                    "" if cpu_percent_avg is None else f"{cpu_percent_avg:.1f}",
                    metrics.get("latency_ms_mean", ""),
                    "" if lat_std is None else f"{lat_std:.2f}",
                    "" if lat_min is None else f"{lat_min:.2f}",
                    "" if lat_max is None else f"{lat_max:.2f}",
                    metrics.get("latency_ms_p50", ""),
                    metrics.get("latency_ms_p95", ""),
                    "" if lat_cv is None else f"{lat_cv:.4f}",
                    metrics.get("throughput_req_per_s", ""),
                    "" if total is None else total,
                    "" if avg is None else f"{avg:.4f}",
                    "" if total_in is None else total_in,
                    "" if total_out is None else total_out,
                    "" if ttft_mean is None else f"{ttft_mean:.2f}",
                    "" if gen_mean is None else f"{gen_mean:.2f}",
                    metrics.get("concurrency", ""),
                    "" if metrics.get("avg_rouge1_f") is None else f"{metrics['avg_rouge1_f']:.4f}",
                    "" if metrics.get("avg_rouge2_f") is None else f"{metrics['avg_rouge2_f']:.4f}",
                    "" if metrics.get("avg_rougeL_f") is None else f"{metrics['avg_rougeL_f']:.4f}",
                ]
                writer.writerow(row)
            except Exception as e:
                print(f"Warning: skipping {run_dir.name}: {e}", file=sys.stderr)
                continue
    finally:
        if close_after:
            out_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())