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

"""NumPy reference forward pass for GPT-2 (debugger / oracle).

Reads the CSV + MTD weights produced by ``convert_gpt2.py`` (indexed by the
sibling ``manifest.json``) and runs a pure-NumPy GPT-2 forward pass.  The main
purpose is *debugging the DML implementation*: the oracle dumps every
intermediate hidden state (after embed, after each block's LN1/attn/LN2/output,
after the final LN, and the logits) so a later comparison harness can
pinpoint exactly which sublayer diverges.

Secondary purpose: independently validate the converter against the original
HuggingFace model without DML in the loop (``--compare-hf``).

Numerical conventions
---------------------
* All math runs in float64, matching DML's native value type.  Source CSVs
  written by the converter are also float64 by default.
* GELU uses the tanh approximation (``gelu_new``), which matches HF's GPT-2
  and the formula in ``scripts/nn/layers/gelu.dml``.
* Attention scaling is by ``sqrt(d_head)`` (per-head dim), not ``sqrt(D)``.
* The LM head is tied to the token embedding: ``logits = h_final @ wte.T``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Manifest / weight loading
# ---------------------------------------------------------------------------

def _load_manifest(weights_dir: Path) -> dict:
    with open(weights_dir / "manifest.json") as f:
        return json.load(f)


def _load_csv(weights_dir: Path, rel: str) -> np.ndarray:
    """Load a converter-emitted CSV as a 2-D float64 ndarray."""
    return np.loadtxt(weights_dir / rel, delimiter=",", dtype=np.float64, ndmin=2)


def _load_all(weights_dir: Path, manifest: dict) -> dict[str, np.ndarray]:
    """Eagerly load every matrix referenced by the manifest."""
    return {name: _load_csv(weights_dir, rel) for name, rel in manifest["files"].items()}


# ---------------------------------------------------------------------------
# Operators (match scripts/nn/layers/*.dml semantically)
# ---------------------------------------------------------------------------

def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    """Per-token LayerNorm; gamma/beta are stored as (1, D) by the converter."""
    g = gamma.reshape(-1)
    b = beta.reshape(-1)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * g + b


def _gelu_new(x: np.ndarray) -> np.ndarray:
    """tanh approximation, matches HF gelu_new and scripts/nn/layers/gelu.dml."""
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _causal_self_attention(
    x: np.ndarray,
    W_Q: np.ndarray, b_Q: np.ndarray,
    W_K: np.ndarray, b_K: np.ndarray,
    W_V: np.ndarray, b_V: np.ndarray,
    W_o: np.ndarray, b_o: np.ndarray,
    n_head: int,
) -> np.ndarray:
    """GPT-2 multi-head causal self-attention. Input/output shape: (T, D)."""
    T, D = x.shape
    d = D // n_head

    # Project. Biases stored as (1, D); broadcast on the row axis.
    Q = x @ W_Q + b_Q.reshape(-1)
    K = x @ W_K + b_K.reshape(-1)
    V = x @ W_V + b_V.reshape(-1)

    # (T, D) -> (H, T, d): split last axis into heads, then move heads to front.
    def split_heads(t: np.ndarray) -> np.ndarray:
        return t.reshape(T, n_head, d).transpose(1, 0, 2)

    Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

    # Scaled dot-product per head: (H, T, T).
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(d)

    # Causal mask: zero out j > i (future tokens).
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    # Numerically stable softmax along the key (last) axis.
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)

    ctx = probs @ Vh                                # (H, T, d)
    ctx = ctx.transpose(1, 0, 2).reshape(T, D)      # (T, D)

    return ctx @ W_o + b_o.reshape(-1)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(
    token_ids: Sequence[int],
    weights_dir: str | Path,
    dump: bool = True,
) -> dict[str, np.ndarray]:
    """Run a single forward pass over ``token_ids``.

    Returns a dict of intermediate tensors plus ``logits`` of shape (T, V).
    Keys:
      - ``embed``                       : (T, D) wte+wpe
      - ``h{i}_ln1``                    : (T, D) post-LN1, pre-attn
      - ``h{i}_attn``                   : (T, D) post attention sub-block
                                          (residual already added)
      - ``h{i}_ln2``                    : (T, D) post-LN2, pre-MLP
      - ``h{i}_out``                    : (T, D) post MLP sub-block (block out)
      - ``lnf``                         : (T, D) post final LayerNorm
      - ``logits``                      : (T, V)
    """
    weights_dir = Path(weights_dir)
    manifest = _load_manifest(weights_dir)
    cfg = manifest["config"]
    D, H, n_layer = cfg["n_embd"], cfg["n_head"], cfg["n_layer"]
    n_ctx, vocab = cfg["n_ctx"], cfg["vocab_size"]
    eps = float(cfg["layer_norm_eps"])

    ids = np.asarray(list(token_ids), dtype=np.int64)
    T = ids.shape[0]
    if T > n_ctx:
        raise ValueError(f"sequence length {T} exceeds n_ctx {n_ctx}")
    if (ids < 0).any() or (ids >= vocab).any():
        raise ValueError(f"token ids out of range [0, {vocab})")

    W = _load_all(weights_dir, manifest)
    wte = W["wte"]                                   # (V, D)
    wpe = W["wpe"]                                   # (n_ctx, D)

    states: dict[str, np.ndarray] = {}

    # Embedding lookup: positions 0..T-1.
    h = wte[ids] + wpe[np.arange(T)]
    if dump:
        states["embed"] = h.copy()

    # Pre-LN transformer blocks.
    for i in range(n_layer):
        # --- Attention sub-block ---
        ln1 = _layer_norm(h, W[f"h{i}_ln1_gamma"], W[f"h{i}_ln1_beta"], eps)
        if dump:
            states[f"h{i}_ln1"] = ln1.copy()

        attn_out = _causal_self_attention(
            ln1,
            W[f"h{i}_W_Q"], W[f"h{i}_b_Q"],
            W[f"h{i}_W_K"], W[f"h{i}_b_K"],
            W[f"h{i}_W_V"], W[f"h{i}_b_V"],
            W[f"h{i}_W_context"], W[f"h{i}_b_context"],
            n_head=H,
        )
        h = h + attn_out
        if dump:
            states[f"h{i}_attn"] = h.copy()

        # --- MLP sub-block ---
        ln2 = _layer_norm(h, W[f"h{i}_ln2_gamma"], W[f"h{i}_ln2_beta"], eps)
        if dump:
            states[f"h{i}_ln2"] = ln2.copy()

        mlp_hidden = ln2 @ W[f"h{i}_W_intermediate"] + W[f"h{i}_b_intermediate"].reshape(-1)
        mlp_hidden = _gelu_new(mlp_hidden)
        mlp_out = mlp_hidden @ W[f"h{i}_W_out"] + W[f"h{i}_b_out"].reshape(-1)
        h = h + mlp_out
        if dump:
            states[f"h{i}_out"] = h.copy()

    # Final LayerNorm + tied LM head.
    h = _layer_norm(h, W["lnf_gamma"], W["lnf_beta"], eps)
    if dump:
        states["lnf"] = h.copy()

    logits = h @ wte.T                               # (T, V)
    states["logits"] = logits
    return states


# ---------------------------------------------------------------------------
# Optional: cross-check against HuggingFace
# ---------------------------------------------------------------------------

def compare_with_hf(
    states: dict[str, np.ndarray],
    token_ids: Sequence[int],
    model_id: str,
    atol: float = 1e-4,
    use_float64: bool = True,
) -> dict[str, float]:
    """Run HF on the same tokens, return per-step max-abs-diff vs ``states``.

    HF's ``output_hidden_states`` returns ``n_layer + 1`` tensors:
      idx 0           : post-embedding (matches our ``embed``)
      idx 1..n_layer-1: input to block i (= output of block i-1, matches
                        our ``h{i-1}_out``)
      idx n_layer     : post-final-LayerNorm (matches our ``lnf``)
    The pre-``ln_f`` output of the last block is *not* exposed by HF, so we
    skip ``h{n_layer-1}_out`` here and rely on ``lnf`` and ``logits`` instead.

    By default we upcast HF to float64 -- otherwise per-block diffs of ~3e-3
    are pure float32 quantization noise, not a real correctness signal.
    """
    try:
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError as e:
        raise SystemExit(
            "ERROR: torch + transformers required for --compare-hf"
        ) from e

    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    model = GPT2LMHeadModel.from_pretrained(model_id)
    if use_float64:
        model = model.double()
    model.eval()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    hf_hidden = [h[0].numpy().astype(np.float64) for h in out.hidden_states]
    hf_logits = out.logits[0].numpy().astype(np.float64)
    n_layer = len(hf_hidden) - 1

    diffs: dict[str, float] = {}
    diffs["embed"] = float(np.abs(states["embed"] - hf_hidden[0]).max())
    for i in range(n_layer - 1):
        key = f"h{i}_out"
        if key in states:
            diffs[key] = float(np.abs(states[key] - hf_hidden[i + 1]).max())
    if "lnf" in states:
        diffs["lnf"] = float(np.abs(states["lnf"] - hf_hidden[n_layer]).max())
    diffs["logits"] = float(np.abs(states["logits"] - hf_logits).max())

    precision = "float64" if use_float64 else "float32"
    print(f"[oracle] HF cross-check (atol={atol}, hf={precision}):", file=sys.stderr)
    worst = 0.0
    for k, v in diffs.items():
        flag = "OK " if v <= atol else "FAIL"
        worst = max(worst, v)
        print(f"  {flag}  {k:>10s}  max|d| = {v:.3e}", file=sys.stderr)
    print(f"  worst = {worst:.3e}", file=sys.stderr)
    return diffs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_tokens(spec: str) -> list[int]:
    """Parse '--tokens' as comma-separated ints or @path/to/file (one per line)."""
    if spec.startswith("@"):
        return [int(x) for x in Path(spec[1:]).read_text().split() if x]
    return [int(x) for x in spec.split(",") if x.strip()]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="np_oracle_gpt2",
        description="Pure-NumPy GPT-2 forward pass over converted weights.",
    )
    p.add_argument("--weights", required=True,
                   help="directory containing manifest.json + *.csv from convert_gpt2.py")
    p.add_argument("--tokens", default="464,2068,7586,21831,625,262,16931,3290,13",
                   help="comma-separated token ids, or @file with whitespace-separated ids "
                        "(default: small fixed prompt for smoke testing)")
    p.add_argument("--dump", default=None,
                   help="if set, write all intermediate states to <dump>/states.npz")
    p.add_argument("--compare-hf", action="store_true",
                   help="cross-check logits + hidden states against HuggingFace")
    p.add_argument("--atol", type=float, default=1e-4,
                   help="tolerance for --compare-hf (default: 1e-4)")
    p.add_argument("--hf-float32", action="store_true",
                   help="run HF in float32 (default: cast HF to float64 to match oracle)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)
    weights = Path(args.weights)
    ids = _parse_tokens(args.tokens)
    print(f"[oracle] forward over T={len(ids)} tokens from {weights}", file=sys.stderr)

    states = forward(ids, weights, dump=True)

    print(f"[oracle] logits shape = {states['logits'].shape} "
          f"argmax(last) = {int(states['logits'][-1].argmax())}", file=sys.stderr)

    if args.dump:
        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "states.npz", **states, token_ids=np.asarray(ids, dtype=np.int64))
        print(f"[oracle] wrote {len(states)} arrays to {out / 'states.npz'}", file=sys.stderr)

    if args.compare_hf:
        manifest = _load_manifest(weights)
        compare_with_hf(states, ids, manifest["model"],
                        atol=args.atol, use_float64=not args.hf_float32)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
