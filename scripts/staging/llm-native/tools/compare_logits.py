# -------------------------------------------------------------
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
# -------------------------------------------------------------

"""Three-way correctness check: HuggingFace ↔ NumPy oracle ↔ DML driver.

This is the single runnable artifact that demonstrates per-layer numerical
parity between the three implementations of GPT-2 forward in this project:

  HuggingFace (PyTorch, float64)
        │   reference; ground truth in the field
        ▼
  NumPy oracle  (tools/np_oracle_gpt2.py)
        │   reads converter CSVs; pure NumPy; no DML in the loop
        ▼
  DML driver    (dml/gpt2_inference.dml)
        │   reads stacked CSVs; SystemDS native execution

Default mode is *lightweight*: tokenize, run HF + oracle, print per-step
max-abs-diff.  Use ``--with-dml`` to additionally run the DML driver via
subprocess and add the Oracle↔DML and HF↔DML columns; this takes ~10s on
gpt2-small instead of <1s.

Usage examples
--------------

  # Quick sanity check (HF vs oracle only):
  python tools/compare_logits.py "Hello, my name is"

  # Full three-way check (also runs DML):
  python tools/compare_logits.py --with-dml "Hello, my name is"

  # Longer-prompt sanity (does parity hold at T=128?):
  python tools/compare_logits.py --with-dml --tmax 128 "$(cat long_prompt.txt)"

  # Machine-readable JSON for CI:
  python tools/compare_logits.py --with-dml --json "Hello, my name is"

Exit code is 0 iff every measured max-abs-diff is ≤ ``--tolerance``
(default 1e-9, comfortably above float64 round-off).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np

# Reuse the oracle's pure-NumPy forward; keeps math centralized.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_oracle_gpt2 import forward as oracle_forward  # noqa: E402


# Keys whose dumps the DML driver writes; everything else (h{i}_ln1, h{i}_attn,
# h{i}_ln2) is internal to gpt2_layer::forward and only the oracle dumps it.
_DML_KEYS = ["embed"] + [f"h{i}_out" for i in range(12)] + ["lnf", "logits"]


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _tokenize(prompt: str, model_id: str, tmax: int | None) -> list[int]:
    try:
        from transformers import GPT2TokenizerFast
    except ImportError as e:
        raise SystemExit(
            "ERROR: transformers required.  pip install -r requirements.txt"
        ) from e
    tok = GPT2TokenizerFast.from_pretrained(model_id)
    ids = tok.encode(prompt)
    if tmax is not None and len(ids) > tmax:
        ids = ids[:tmax]
    if len(ids) == 0:
        raise SystemExit("ERROR: empty prompt yields zero tokens")
    return ids


# ---------------------------------------------------------------------------
# Reference runs
# ---------------------------------------------------------------------------

def _run_hf(token_ids: Sequence[int], model_id: str) -> tuple[dict[str, np.ndarray], float]:
    """HF forward in float64; returns dumps in oracle's key convention."""
    try:
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError as e:
        raise SystemExit("ERROR: torch + transformers required") from e

    t0 = time.time()
    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    model = GPT2LMHeadModel.from_pretrained(model_id).double().eval()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    elapsed = time.time() - t0

    hf_hidden = [h[0].numpy().astype(np.float64) for h in out.hidden_states]
    n_layer = len(hf_hidden) - 1

    # HF mapping (see np_oracle_gpt2.compare_with_hf for derivation):
    #   hidden_states[0]            == post-embed       (oracle 'embed')
    #   hidden_states[1..n_layer-1] == output of block i (oracle 'h{i}_out')
    #   hidden_states[n_layer]      == post-final-LN    (oracle 'lnf')
    dumps: dict[str, np.ndarray] = {"embed": hf_hidden[0]}
    for i in range(n_layer - 1):
        dumps[f"h{i}_out"] = hf_hidden[i + 1]
    dumps["lnf"] = hf_hidden[n_layer]
    dumps["logits"] = out.logits[0].numpy().astype(np.float64)
    # HF doesn't expose the pre-ln_f output of the last block, so 'h{n_layer-1}_out'
    # is intentionally absent from the HF dumps.
    return dumps, elapsed


def _run_oracle(token_ids: Sequence[int], weights_dir: Path) -> tuple[dict[str, np.ndarray], float]:
    t0 = time.time()
    states = oracle_forward(token_ids, weights_dir, dump=True)
    return states, time.time() - t0


def _run_dml(
    token_ids: Sequence[int],
    weights_dir: Path,
    systemds_bin: Path,
    systemds_root: Path,
    keep_tmp: bool = False,
) -> tuple[dict[str, np.ndarray], float, Path]:
    """Invoke the DML driver via subprocess, return its on-disk dumps."""
    if not systemds_bin.exists():
        raise SystemExit(f"ERROR: systemds binary not found at {systemds_bin}")

    tmp = Path(tempfile.mkdtemp(prefix="gpt2_compare_"))
    tokens_path = tmp / "tokens.csv"
    dump_dir = tmp / "dml_dumps"

    # Write tokens (one id per line) plus the matching .mtd so SystemDS picks
    # up the (T, 1) double-matrix shape without inference.
    # Filename intentionally does not start with '_' or '.': Hadoop's
    # FileInputFormat (used for CSV reads) silently skips files matching
    # those prefixes as hidden / partition markers, surfacing only as a
    # cryptic "Input path does not exist" at DML runtime.
    tokens_path.write_text("\n".join(str(int(i)) for i in token_ids) + "\n")
    mtd = {
        "data_type": "matrix", "value_type": "double", "format": "csv",
        "header": False, "sep": ",",
        "rows": len(token_ids), "cols": 1,
    }
    (tokens_path.with_suffix(".csv.mtd")).write_text(json.dumps(mtd))

    driver = systemds_root / "scripts/staging/llm-native/dml/gpt2_inference.dml"

    env = dict(os.environ)
    env["SYSTEMDS_ROOT"] = str(systemds_root)
    env["SYSDS_QUIET"] = "1"

    cmd = [
        str(systemds_bin), str(driver),
        "-nvargs",
        f"weights={weights_dir}",
        f"tokens={tokens_path}",
        f"out={dump_dir}",
        "dump=TRUE",
    ]

    # cwd must be the repo root: the driver source()'s nn/layers/* via paths
    # that resolve against the SystemDS jar's bundled scripts/ tree, which
    # only takes precedence when launched from there.
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(systemds_root),
                          capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"ERROR: DML driver exited {proc.returncode}")

    dumps: dict[str, np.ndarray] = {}
    for k in _DML_KEYS:
        path = dump_dir / f"{k}.csv"
        if not path.exists():
            sys.stderr.write("[compare] --- DML stdout ---\n")
            sys.stderr.write(proc.stdout)
            sys.stderr.write("[compare] --- DML stderr ---\n")
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"ERROR: DML driver did not produce {path}")
        dumps[k] = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)

    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return dumps, elapsed, tmp


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).max())


def _build_report(
    hf: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
    dml: dict[str, np.ndarray] | None,
) -> dict:
    """Return per-step diffs across whatever pairs are available."""
    rows: list[dict] = []
    # Oracle has the most keys (h{i}_ln1/attn/ln2 too); we only report on
    # rows that exist in *every* available source so the table is rectangular.
    base_keys = ["embed"] + [f"h{i}_out" for i in range(12)] + ["lnf", "logits"]

    for k in base_keys:
        row: dict = {"key": k, "shape": list(oracle[k].shape) if k in oracle else None}
        if k in hf and k in oracle:
            row["hf_vs_oracle"] = _max_abs_diff(hf[k], oracle[k])
        if dml is not None and k in dml and k in oracle:
            row["oracle_vs_dml"] = _max_abs_diff(oracle[k], dml[k])
        if dml is not None and k in dml and k in hf:
            row["hf_vs_dml"] = _max_abs_diff(hf[k], dml[k])
        rows.append(row)
    return {"rows": rows}


def _print_table(report: dict, with_dml: bool) -> float:
    """Pretty table; returns the worst diff seen anywhere."""
    cols = ["HF vs Oracle"]
    if with_dml:
        cols += ["Oracle vs DML", "HF vs DML"]
    header = f"  {'step':<10s} | " + " | ".join(f"{c:>14s}" for c in cols)
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    worst = 0.0

    def fmt(v):
        if v is None:
            return f"{'-':>14s}"
        return f"{v:>14.3e}"

    for row in report["rows"]:
        cells = [fmt(row.get("hf_vs_oracle"))]
        if with_dml:
            cells.append(fmt(row.get("oracle_vs_dml")))
            cells.append(fmt(row.get("hf_vs_dml")))
        for k in ("hf_vs_oracle", "oracle_vs_dml", "hf_vs_dml"):
            v = row.get(k)
            if v is not None and not np.isnan(v):
                worst = max(worst, v)
        print(f"  {row['key']:<10s} | " + " | ".join(cells))
    print(sep)
    print(f"  {'worst':<10s} | {worst:>14.3e}")
    return worst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_logits",
        description="HF vs NumPy oracle vs DML driver -- per-layer parity check.",
    )
    p.add_argument("prompt", nargs="?", default="Hello, my name is",
                   help="text prompt to encode (default: 'Hello, my name is')")
    p.add_argument("--weights", default=None,
                   help="converted weights dir (default: scripts/staging/llm-native/weights/gpt2)")
    p.add_argument("--tmax", type=int, default=16,
                   help="truncate tokenization to first N tokens (default: 16)")
    p.add_argument("--tolerance", type=float, default=1e-9,
                   help="max-abs-diff threshold for PASS (default: 1e-9)")
    p.add_argument("--with-dml", action="store_true",
                   help="also run the DML driver and add Oracle/DML and HF/DML columns")
    p.add_argument("--systemds-bin", default=None,
                   help="path to bin/systemds (default: $SYSTEMDS_ROOT/bin/systemds)")
    p.add_argument("--systemds-root", default=None,
                   help="repo root (default: walk up from this script)")
    p.add_argument("--keep-tmp", action="store_true",
                   help="keep DML temp dir (useful for debugging diffs)")
    p.add_argument("--json", action="store_true",
                   help="machine-readable JSON output instead of pretty table")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)

    here = Path(__file__).resolve()
    repo_root = Path(args.systemds_root) if args.systemds_root else here.parents[4]
    weights = Path(args.weights) if args.weights \
        else repo_root / "scripts/staging/llm-native/weights/gpt2"
    if not (weights / "manifest.json").exists():
        raise SystemExit(f"ERROR: no manifest.json under {weights} -- run convert_gpt2.py first")

    with open(weights / "manifest.json") as f:
        manifest = json.load(f)
    model_id = manifest["model"]

    ids = _tokenize(args.prompt, model_id, args.tmax)
    if not args.json:
        print(f"[compare] model={model_id}  T={len(ids)}  prompt={args.prompt!r}",
              file=sys.stderr)

    hf_dumps,  hf_t = _run_hf(ids, model_id)
    or_dumps,  or_t = _run_oracle(ids, weights)

    dml_dumps = None
    dml_t = 0.0
    dml_tmp: Path | None = None
    if args.with_dml:
        if args.with_dml and "stacked" not in manifest:
            raise SystemExit(
                f"ERROR: {weights}/manifest.json has no 'stacked' entry -- "
                "run pack_weights.py before --with-dml"
            )
        sysds_bin = Path(args.systemds_bin) if args.systemds_bin \
            else repo_root / "bin/systemds"
        dml_dumps, dml_t, dml_tmp = _run_dml(ids, weights, sysds_bin, repo_root,
                                             keep_tmp=args.keep_tmp)

    report = _build_report(hf_dumps, or_dumps, dml_dumps)
    report["meta"] = {
        "model": model_id,
        "prompt": args.prompt,
        "T": len(ids),
        "tolerance": args.tolerance,
        "elapsed_sec": {"hf": hf_t, "oracle": or_t, "dml": dml_t},
    }

    if args.json:
        print(json.dumps(report, indent=2))
        worst = max(
            (v for row in report["rows"]
             for k, v in row.items() if k in ("hf_vs_oracle", "oracle_vs_dml", "hf_vs_dml")
             and v is not None and not (isinstance(v, float) and np.isnan(v))),
            default=0.0,
        )
    else:
        worst = _print_table(report, with_dml=args.with_dml)
        print()
        print(f"  HF      run: {hf_t:6.2f} s")
        print(f"  Oracle  run: {or_t:6.2f} s")
        if args.with_dml:
            print(f"  DML     run: {dml_t:6.2f} s")
        if dml_tmp is not None and args.keep_tmp:
            print(f"  DML  tmp dir: {dml_tmp}")

    ok = worst <= args.tolerance
    if not args.json:
        verdict = "PASS" if ok else "FAIL"
        print(f"\n  {verdict}: worst {worst:.3e} {'≤' if ok else '>'} tol {args.tolerance:.1e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
