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

"""Convert a HuggingFace GPT-2 checkpoint to SystemDS CSV + MTD format.

Each parameter tensor is written as a `<name>.csv` file (comma-separated,
float64 by default) accompanied by a `<name>.csv.mtd` JSON metadata file that
SystemDS uses to type the matrix.  A top-level `manifest.json` records the
model config, file map, tied weights (LM head), and SHA-256 hashes.

The shape conventions follow `scripts/nn/layers/affine.dml` and
`scripts/nn/layers/bert_layer.dml`:

  - 2-D weight matrices keep their HF shape (D, M); HF's `Conv1D` already
    stores weights as (in, out), matching DML's affine `W : (D, M)`.
  - 1-D bias / LayerNorm vectors are reshaped from (D,) to (1, D).
  - Combined `attn.c_attn.weight` of shape (D, 3D) is column-sliced into
    three (D, D) tensors W_Q, W_K, W_V; same for the bias vector.
  - `lm_head.weight` is tied to `wte.weight` and recorded in the manifest
    instead of being duplicated on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Iterable

import numpy as np


_MTD_TEMPLATE = {
    "data_type": "matrix",
    "value_type": "double",
    "format": "csv",
    "header": False,
    "sep": ",",
}


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """SystemDS matrices are 2-D; promote 1-D vectors to (1, N) row vectors."""
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"unsupported tensor rank {arr.ndim}; only 1-D/2-D supported")


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_matrix(out_dir: str, name: str, arr: np.ndarray, dtype: str) -> tuple[str, tuple[int, int]]:
    """Write `<out_dir>/<name>.csv` and `<name>.csv.mtd`. Returns (relpath, shape)."""
    arr = _ensure_2d(np.asarray(arr))
    if dtype == "float64":
        arr = arr.astype(np.float64, copy=False)
        value_type = "double"
    elif dtype == "float32":
        arr = arr.astype(np.float32, copy=False)
        value_type = "single"
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    rel = f"{name}.csv"
    csv_path = os.path.join(out_dir, rel)
    mtd_path = csv_path + ".mtd"

    fmt = "%.17g" if dtype == "float64" else "%.9g"
    np.savetxt(csv_path, arr, fmt=fmt, delimiter=",")

    mtd = dict(_MTD_TEMPLATE)
    mtd["value_type"] = value_type
    mtd["rows"] = int(arr.shape[0])
    mtd["cols"] = int(arr.shape[1])
    with open(mtd_path, "w") as f:
        json.dump(mtd, f, indent=2)

    return rel, (int(arr.shape[0]), int(arr.shape[1]))


def _to_numpy(t):
    """Detach a torch tensor to a contiguous CPU NumPy array (passthrough for ndarrays)."""
    if isinstance(t, np.ndarray):
        return t
    return t.detach().to("cpu").contiguous().numpy()


def convert(
    model_id: str,
    out_dir: str,
    dtype: str = "float64",
    cache_dir: str | None = None,
) -> dict:
    """Convert an HF GPT-2 checkpoint.  Returns the in-memory manifest dict."""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as e:
        raise SystemExit(
            "ERROR: transformers is required.  pip install -r requirements.txt"
        ) from e

    os.makedirs(out_dir, exist_ok=True)

    print(f"[convert_gpt2] loading {model_id} ...", file=sys.stderr)
    model = GPT2LMHeadModel.from_pretrained(model_id, cache_dir=cache_dir)
    model.eval()

    # GPT2LMHeadModel.state_dict() prefixes inner-module keys with "transformer."
    # and exposes lm_head.weight (which is tied to wte.weight).  Use the inner
    # GPT2Model directly so keys read like "wte.weight", "h.0.attn.c_attn.weight".
    sd = model.transformer.state_dict()
    cfg = model.config
    D = cfg.n_embd
    n_layer = cfg.n_layer
    activation = getattr(cfg, "activation_function", "gelu_new")

    files: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}

    def emit(name: str, tensor) -> None:
        rel, shape = _write_matrix(out_dir, name, _to_numpy(tensor), dtype)
        files[name] = rel
        shapes[name] = list(shape)

    # Embeddings (used outside the per-block forward).
    emit("wte", sd["wte.weight"])
    emit("wpe", sd["wpe.weight"])

    for i in range(n_layer):
        # LayerNorm 1.
        emit(f"h{i}_ln1_gamma", sd[f"h.{i}.ln_1.weight"])
        emit(f"h{i}_ln1_beta",  sd[f"h.{i}.ln_1.bias"])

        # Combined Q|K|V projection: split along the output (column) axis.
        Wc = _to_numpy(sd[f"h.{i}.attn.c_attn.weight"])  # (D, 3D)
        bc = _to_numpy(sd[f"h.{i}.attn.c_attn.bias"])    # (3D,)
        if Wc.shape != (D, 3 * D):
            raise RuntimeError(
                f"unexpected c_attn.weight shape {Wc.shape}; expected ({D}, {3 * D})"
            )
        emit(f"h{i}_W_Q", Wc[:,        0:    D])
        emit(f"h{i}_W_K", Wc[:,        D:2 * D])
        emit(f"h{i}_W_V", Wc[:,    2 * D:3 * D])
        emit(f"h{i}_b_Q", bc[          0:    D])
        emit(f"h{i}_b_K", bc[          D:2 * D])
        emit(f"h{i}_b_V", bc[      2 * D:3 * D])

        # Attention output projection (DML calls this W_context / b_context).
        emit(f"h{i}_W_context", sd[f"h.{i}.attn.c_proj.weight"])
        emit(f"h{i}_b_context", sd[f"h.{i}.attn.c_proj.bias"])

        # LayerNorm 2.
        emit(f"h{i}_ln2_gamma", sd[f"h.{i}.ln_2.weight"])
        emit(f"h{i}_ln2_beta",  sd[f"h.{i}.ln_2.bias"])

        # Feed-forward (MLP).
        emit(f"h{i}_W_intermediate", sd[f"h.{i}.mlp.c_fc.weight"])
        emit(f"h{i}_b_intermediate", sd[f"h.{i}.mlp.c_fc.bias"])
        emit(f"h{i}_W_out",          sd[f"h.{i}.mlp.c_proj.weight"])
        emit(f"h{i}_b_out",          sd[f"h.{i}.mlp.c_proj.bias"])

    # Final LayerNorm.
    emit("lnf_gamma", sd["ln_f.weight"])
    emit("lnf_beta",  sd["ln_f.bias"])

    # Compute SHA-256 hashes after all files are flushed.
    hashes = {rel: _sha256_of(os.path.join(out_dir, rel)) for rel in files.values()}

    manifest = {
        "model": model_id,
        "arch": "gpt2-causal",
        "dtype": dtype,
        "config": {
            "n_layer": int(cfg.n_layer),
            "n_head": int(cfg.n_head),
            "n_embd": int(cfg.n_embd),
            "n_ctx": int(cfg.n_ctx),
            "vocab_size": int(cfg.vocab_size),
            "activation": str(activation),
            "layer_norm_eps": float(getattr(cfg, "layer_norm_epsilon", 1e-5)),
        },
        "tied":   {"lm_head": "wte"},
        "files":  files,
        "shapes": shapes,
        "sha256": hashes,
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(
        f"[convert_gpt2] wrote {len(files)} matrices "
        f"(+ manifest) to {out_dir}",
        file=sys.stderr,
    )
    return manifest


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_gpt2",
        description="Convert a HuggingFace GPT-2 checkpoint to SystemDS CSV + MTD.",
    )
    p.add_argument("--model", default="gpt2",
                   help="HuggingFace model id or local path (default: gpt2)")
    p.add_argument("--out", default=None,
                   help="output directory (default: weights/<basename(model)>)")
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64",
                   help="numeric type to write (default: float64, matches DML)")
    p.add_argument("--cache", default=None,
                   help="HuggingFace cache directory (default: HF default)")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)
    out = args.out or os.path.join("weights", os.path.basename(args.model.rstrip("/")))
    convert(model_id=args.model, out_dir=out, dtype=args.dtype, cache_dir=args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
