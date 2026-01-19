
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_run_dir(p: Path) -> bool:
    return p.is_dir() and (p / "metrics.json").exists() and (p / "run_config.json").exists()


def iter_run_dirs(results_dir: Path) -> Iterable[Path]:
    """
    Yields run directories that contain metrics.json and run_config.json.

    Supports:
      results/run_xxx/
      results/<group>/run_xxx/   (one-level nesting)
    Avoids duplicates by tracking resolved paths.
    """
    if not results_dir.exists():
        return

    seen = set()

    # direct children
    for p in results_dir.iterdir():
        if is_run_dir(p):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                yield p

    # one level nesting
    for group in results_dir.iterdir():
        if not group.is_dir():
            continue
        for p in group.iterdir():
            if is_run_dir(p):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    yield p


def manifest_timestamp(run_dir: Path) -> str:
    """
    Returns timestamp_utc string from manifest.json if present; else "".
    Kept as ISO8601 string so CSV stays simple.
    """
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return ""
    try:
        m = read_json(mpath)
        ts = m.get("timestamp_utc")
        return "" if ts is None else str(ts)
    except Exception:
        return ""


def token_stats(samples_path: Path) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
    """
    Returns:
      (total_tokens, avg_tokens, total_input_tokens, total_output_tokens)
    If not available: (None, None, None, None)
    """
    if not samples_path.exists():
        return (None, None, None, None)

    total_tokens = 0
    total_in = 0
    total_out = 0
    count = 0
    saw_any = False

    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                usage = (obj.get("extra") or {}).get("usage") or {}
                tt = usage.get("total_tokens")
                it = usage.get("input_tokens")
                ot = usage.get("output_tokens")

                if tt is None and it is None and ot is None:
                    continue

                saw_any = True
                if tt is not None:
                    total_tokens += int(tt)
                if it is not None:
                    total_in += int(it)
                if ot is not None:
                    total_out += int(ot)

                count += 1
    except Exception:
        return (None, None, None, None)

    if not saw_any or count == 0:
        return (None, None, None, None)

    avg = (total_tokens / count) if total_tokens > 0 else None
    return (
        total_tokens if total_tokens > 0 else None,
        avg,
        total_in if total_in > 0 else None,
        total_out if total_out > 0 else None,
    )


def ttft_stats(samples_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns:
      (ttft_ms_mean, generation_ms_mean)
    If not available: (None, None)
    
    Only processes samples that have TTFT metrics (streaming mode).
    Non-streaming samples are ignored, not treated as zeros.
    
    Checks both top-level and extra dict for backward compatibility.
    """
    if not samples_path.exists():
        return (None, None)

    total_ttft = 0.0
    total_gen = 0.0
    count = 0

    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                # check top level first (new format), then extra dict (backward compat)
                ttft = obj.get("ttft_ms")
                gen = obj.get("generation_ms")
                
                if ttft is None:
                    # fall back to extra dict
                    extra = obj.get("extra") or {}
                    ttft = extra.get("ttft_ms")
                    gen = extra.get("generation_ms")

                # only count samples that have TTFT metrics
                if ttft is not None:
                    total_ttft += float(ttft)
                    if gen is not None:
                        total_gen += float(gen)
                    count += 1

    except Exception:
        return (None, None)

    if count == 0:
        return (None, None)

    return (
        total_ttft / count,
        total_gen / count if total_gen > 0 else None,
    )

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