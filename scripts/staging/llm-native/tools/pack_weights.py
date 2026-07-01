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

"""Pack per-layer GPT-2 weight CSVs into stacked CSVs for the DML driver.

SystemDS' DML parser requires ``read()`` filenames to be const-string-
traceable, which precludes loop-variable filename construction.  The DML
driver therefore reads one stacked tensor per parameter type and row-slices
it inside the per-layer loop.

For each parameter type ``P`` that appears once per layer, this script
emits ``all_<P>.csv`` (+ ``.csv.mtd``) of shape:

  - ``(n_layer * rows_per_layer, cols_per_layer)`` for 2-D matrices
  - ``(n_layer, dim)``                              for row-vector biases
                                                    (already 1xD per layer)

The original per-layer files are untouched -- they remain the canonical
artifact for the NumPy oracle, the converter tests, and human inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


# Per-layer parameter names emitted by ``convert_gpt2.py``.  Keep in sync
# with the ``hi_*`` keys it writes; the DML driver expects exactly these
# stacked files.
_PER_LAYER_KEYS = (
    "ln1_gamma", "ln1_beta",
    "W_Q", "b_Q", "W_K", "b_K", "W_V", "b_V",
    "W_context", "b_context",
    "ln2_gamma", "ln2_beta",
    "W_intermediate", "b_intermediate",
    "W_out", "b_out",
)


_MTD_TEMPLATE = {
    "data_type": "matrix",
    "value_type": "double",
    "format": "csv",
    "header": False,
    "sep": ",",
}


def _load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    np.savetxt(path, arr, fmt="%.17g", delimiter=",")
    mtd = dict(_MTD_TEMPLATE)
    mtd["rows"] = int(arr.shape[0])
    mtd["cols"] = int(arr.shape[1])
    with open(path.with_suffix(path.suffix + ".mtd"), "w") as f:
        json.dump(mtd, f, indent=2)


def pack(weights_dir: str | Path) -> dict[str, list[int]]:
    """Pack per-layer CSVs in ``weights_dir`` into ``all_*.csv`` files.

    Returns a dict ``{stacked_name: [rows, cols]}`` of what was written.
    """
    weights_dir = Path(weights_dir)
    with open(weights_dir / "manifest.json") as f:
        manifest = json.load(f)
    n_layer = int(manifest["config"]["n_layer"])

    written: dict[str, list[int]] = {}
    for key in _PER_LAYER_KEYS:
        per_layer = [_load_csv(weights_dir / f"h{i}_{key}.csv") for i in range(n_layer)]

        # Sanity: all layers must agree on shape so vstack is unambiguous.
        s0 = per_layer[0].shape
        for i, t in enumerate(per_layer):
            if t.shape != s0:
                raise RuntimeError(
                    f"shape mismatch in {key}: layer 0 = {s0}, layer {i} = {t.shape}"
                )

        stacked = np.vstack(per_layer)
        out = weights_dir / f"all_{key}.csv"
        _write_csv(out, stacked)
        written[f"all_{key}"] = [int(stacked.shape[0]), int(stacked.shape[1])]
        print(f"[pack_weights] {out.name:<24s} ({stacked.shape[0]:>5d}, "
              f"{stacked.shape[1]:>5d})  from {n_layer} layers of {s0}",
              file=sys.stderr)

    # Store stacked layout in the manifest under a separate key so existing
    # consumers (oracle, tests) are unaffected.
    manifest.setdefault("stacked", {})
    manifest["stacked"]["per_layer_keys"] = list(_PER_LAYER_KEYS)
    manifest["stacked"]["shapes"] = written
    with open(weights_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"[pack_weights] wrote {len(written)} stacked tensors to {weights_dir}",
          file=sys.stderr)
    return written


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pack_weights",
        description="Stack per-layer GPT-2 CSVs into all_*.csv for the DML driver.",
    )
    p.add_argument("--weights", required=True,
                   help="directory containing manifest.json + h{i}_*.csv")
    args = p.parse_args(list(argv) if argv is not None else None)
    pack(args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
