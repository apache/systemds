<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied.  See the License for the specific language governing
permissions and limitations under the License.
{% endcomment %}
-->

# Native LLM inference in DML (work in progress)

This directory contains tooling for running pre-trained transformer language
models natively inside SystemDS, using the existing `scripts/nn/layers/*.dml`
operators (affine, multi-head attention with optional causal mask, layer norm,
GELU, etc.).

The first model targeted is **GPT-2 small (124M)**.

## Layout

```
llm-native/
├── tools/
│   ├── convert_gpt2.py     # HF GPT-2 -> SystemDS CSV + MTD + manifest.json
│   ├── pack_weights.py     # per-layer CSVs -> stacked all_*.csv (for DML driver)
│   ├── np_oracle_gpt2.py   # pure-NumPy reference forward (debugger)
│   └── compare_logits.py   # three-way HF / oracle / DML parity check
├── dml/
│   └── gpt2_inference.dml  # native DML inference driver
├── tests/
│   └── test_convert_gpt2.py
├── weights/                # generated; gitignored
├── requirements.txt
└── README.md
```

## Quick start

```bash
cd scripts/staging/llm-native
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Convert the HF GPT-2 small checkpoint into DML-ready matrices.
python tools/convert_gpt2.py --model gpt2 --out weights/gpt2

# Pack the per-layer matrices into stacked all_*.csv files for DML.
python tools/pack_weights.py --weights weights/gpt2

# (Optional) Cross-check the converter against HuggingFace via the
# pure-NumPy reference forward.  All per-step diffs should be < 1e-11.
python tools/np_oracle_gpt2.py --weights weights/gpt2 --compare-hf

# Run native DML inference (writes logits.csv + per-block dumps).
echo -e "464\n2068\n7586\n21831\n625\n262\n16931\n3290\n13" > weights/gpt2/tokens.csv
SYSTEMDS_ROOT=$PWD/../../.. $SYSTEMDS_ROOT/bin/systemds dml/gpt2_inference.dml \
  -nvargs weights=weights/gpt2 \
          tokens=weights/gpt2/tokens.csv \
          out=weights/gpt2/dml_dumps \
          dump=TRUE

# End-to-end three-way parity check (HF + NumPy oracle + DML driver):
python tools/compare_logits.py --with-dml "Hello, my name is"
```

`compare_logits.py` is the canonical artifact for "does native DML GPT-2
match HuggingFace?".  Default mode runs only HF + oracle (~1s, no DML); pass
`--with-dml` for the full three-way check (~10s; requires the converted
weights *and* a successful `pack_weights.py` run).  All measured per-step
max-abs-diffs at gpt2 (124M) sit at ~1e-12, the float64 round-off floor.

The converter is a one-shot Python script.  After it runs, the `weights/gpt2/`
directory contains one `<name>.csv` plus matching `<name>.csv.mtd` per
parameter tensor, plus a `manifest.json` describing the model config, file
map, tied weights, and SHA-256 hashes.

## Why a converter?

The DML transformer layers in `scripts/nn/layers/` describe **computation only**
(matmuls, layer norm, attention).  Trained parameter values live in HuggingFace
PyTorch checkpoints with HF-specific names and shapes.  The converter is the
one-time bridge:

1. Loads the HF `GPT2LMHeadModel` weights.
2. Splits the fused `c_attn` projection back into separate `W_Q`, `W_K`, `W_V`.
3. Reshapes biases from `(D,)` to `(1, D)` to match DML conventions.
4. Upcasts to `float64` (DML's native value type).
5. Writes one CSV + MTD pair per matrix and a `manifest.json` index.

After conversion, DML scripts can `read("weights/gpt2/h0_W_Q.csv", format="csv")`
and feed the matrices directly to `bert_layer::forward(...)` /
`multi_attention::forward_causal(...)`.

## HF -> DML name and shape mapping

`B`/`T`/`D`/`H`/`I`/`V` follow the conventions of `bert_layer.dml`
(`D = n_embd = 768`, `I = 4*D = 3072`, `V = 50257` for GPT-2 small).

| HF state-dict key              | Shape (HF)     | DML file (this dir)        | DML role             |
|--------------------------------|----------------|----------------------------|----------------------|
| `wte.weight`                   | `(V, D)`       | `wte.csv`                  | token embedding      |
| `wpe.weight`                   | `(n_ctx, D)`   | `wpe.csv`                  | positional embedding |
| `h.i.ln_1.weight`              | `(D,)`         | `hi_ln1_gamma.csv`         | LN1 gamma            |
| `h.i.ln_1.bias`                | `(D,)`         | `hi_ln1_beta.csv`          | LN1 beta             |
| `h.i.attn.c_attn.weight[:,0:D]`| `(D, D)`       | `hi_W_Q.csv`               | query projection W   |
| `h.i.attn.c_attn.weight[:,D:2D]`|`(D, D)`       | `hi_W_K.csv`               | key projection W     |
| `h.i.attn.c_attn.weight[:,2D:3D]`|`(D, D)`      | `hi_W_V.csv`               | value projection W   |
| `h.i.attn.c_attn.bias[0:D]`    | `(D,)`         | `hi_b_Q.csv` (1xD)         | query bias           |
| `h.i.attn.c_attn.bias[D:2D]`   | `(D,)`         | `hi_b_K.csv`               | key bias             |
| `h.i.attn.c_attn.bias[2D:3D]`  | `(D,)`         | `hi_b_V.csv`               | value bias           |
| `h.i.attn.c_proj.weight`       | `(D, D)`       | `hi_W_context.csv`         | attn out W           |
| `h.i.attn.c_proj.bias`         | `(D,)`         | `hi_b_context.csv`         | attn out bias        |
| `h.i.ln_2.weight`              | `(D,)`         | `hi_ln2_gamma.csv`         | LN2 gamma            |
| `h.i.ln_2.bias`                | `(D,)`         | `hi_ln2_beta.csv`          | LN2 beta             |
| `h.i.mlp.c_fc.weight`          | `(D, I)`       | `hi_W_intermediate.csv`    | MLP expand W         |
| `h.i.mlp.c_fc.bias`            | `(I,)`         | `hi_b_intermediate.csv`    | MLP expand bias      |
| `h.i.mlp.c_proj.weight`        | `(I, D)`       | `hi_W_out.csv`             | MLP contract W       |
| `h.i.mlp.c_proj.bias`          | `(D,)`         | `hi_b_out.csv`             | MLP contract bias    |
| `ln_f.weight`                  | `(D,)`         | `lnf_gamma.csv`            | final LN gamma       |
| `ln_f.bias`                    | `(D,)`         | `lnf_beta.csv`             | final LN beta        |
| `lm_head.weight`               | tied to `wte`  | (none)                     | recorded in manifest |

GPT-2 uses HuggingFace's `Conv1D` linear layer, which stores weights as
`(in, out)` -- exactly what the DML affine layer (`W : (D, M)`) expects, so
no transpose is performed during conversion.

## Manifest format

`manifest.json` is the index DML drivers should consult first:

```json
{
  "model": "gpt2",
  "arch": "gpt2-causal",
  "config": {
    "n_layer": 12, "n_head": 12, "n_embd": 768,
    "n_ctx": 1024, "vocab_size": 50257,
    "activation": "gelu", "layer_norm_eps": 1.0e-5
  },
  "dtype": "float64",
  "tied":   { "lm_head": "wte" },
  "files":  { "wte": "wte.csv", "...": "..." },
  "sha256": { "wte.csv": "ab12...", "...": "..." }
}
```

`tied.lm_head = wte` means the DML driver should reuse `wte` (transposed) for
the language-modeling head rather than expecting a separate file.

## CLI

```
python tools/convert_gpt2.py [options]

  --model   HF model id or local path  (default: gpt2)
  --out     output directory            (default: weights/<basename(model)>)
  --dtype   {float64,float32}           (default: float64)
  --cache   HuggingFace cache directory (default: HF default)
```

## Tests

```bash
pytest scripts/staging/llm-native/tests
```

The test suite uses `sshleifer/tiny-gpt2` (a 5-layer 64-dim fixture, a few MB)
to verify the converter end-to-end without downloading the full GPT-2 weights.

## Implementation notes (gotchas worth knowing)

A few SystemDS / Hadoop quirks that shaped the design here, recorded so the
next person who tries this doesn't lose a day to them:

1. **DML `read()` paths must be const-string-traceable.**  The DML parser
   rejects `read()` calls whose filename argument is built from runtime
   variables -- including loop counters and `ifdef` defaults that aren't
   string literals.  Trying to `read("weights/h" + i + "_W_Q.csv", ...)` in a
   loop fails at parse time with a `NullPointerException` deep inside
   `StringIdentifier.getValue()`, which is a deeply unhelpful error.

   Workaround used here: `pack_weights.py` `vstack`s the 12 per-layer copies
   of each parameter into a single `all_<param>.csv` (e.g. `all_W_Q.csv` of
   shape `(12*D, D)`).  The DML driver `read()`s 16 stacked files once, then
   row-slices inside the per-layer loop -- no runtime-built paths needed.
   This also turns 196 disk reads into 16, which is why startup is bearable
   on a laptop.

2. **Hadoop's `FileInputFormat` silently skips files whose names start with
   `_` or `.`** (its hidden / partition-marker convention).  SystemDS uses
   the Hadoop input layer for CSV reads, so a tokens file called
   `_tokens.csv` will produce an `InvalidInputException("Input path does not
   exist")` at runtime even though `ls` shows it sitting right there.  Name
   inputs without a leading underscore.
