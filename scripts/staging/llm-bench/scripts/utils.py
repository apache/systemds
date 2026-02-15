"""Shared utilities for aggregate.py and report.py."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    """Read and parse a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_run_dir(p: Path) -> bool:
    """Check if a directory is a valid benchmark run directory."""
    return p.is_dir() and (p / "metrics.json").exists() and (p / "run_config.json").exists()


def iter_run_dirs(results_dir: Path) -> list:
    """
    Returns run directories that contain metrics.json and run_config.json.

    Supports:
      results/run_xxx/
      results/<group>/run_xxx/   (one-level nesting)
    Avoids duplicates by tracking resolved paths.
    """
    if not results_dir.exists():
        return []

    seen = set()
    runs = []

    # direct children
    for p in results_dir.iterdir():
        if is_run_dir(p):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                runs.append(p)

    # one level nesting
    for group in results_dir.iterdir():
        if not group.is_dir():
            continue
        for p in group.iterdir():
            if is_run_dir(p):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    runs.append(p)

    return runs


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
    ttft_count = 0
    gen_count = 0

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

                # track ttft and gen independently
                if ttft is not None:
                    total_ttft += float(ttft)
                    ttft_count += 1
                if gen is not None:
                    total_gen += float(gen)
                    gen_count += 1

    except Exception:
        return (None, None)

    if ttft_count == 0:
        return (None, None)

    return (
        total_ttft / ttft_count,
        total_gen / gen_count if gen_count > 0 else None,
    )
