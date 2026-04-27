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

"""Tests for tools/convert_gpt2.py.

Uses `sshleifer/tiny-gpt2` (a few MB, 2 layers, 32-dim embedding) so the
suite runs end-to-end without downloading the full GPT-2 weights.

Run with:

    pytest scripts/staging/llm-native/tests
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, os.pardir, "tools"))
sys.path.insert(0, TOOLS)

# Importing transformers/torch is expensive; import lazily inside fixtures.
TINY = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from convert_gpt2 import convert  # noqa: WPS433 (sys.path manipulated above)

    out = tmp_path_factory.mktemp("tiny_gpt2_weights")
    manifest = convert(model_id=TINY, out_dir=str(out), dtype="float64")
    return out, manifest


def _load_csv(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


def test_manifest_top_level(converted):
    out, manifest = converted
    assert manifest["arch"] == "gpt2-causal"
    assert manifest["dtype"] == "float64"
    assert manifest["tied"] == {"lm_head": "wte"}
    assert manifest["model"] == TINY
    cfg = manifest["config"]
    assert cfg["n_layer"] >= 1
    assert cfg["n_embd"] >= 1
    assert cfg["vocab_size"] > 0
    assert (out / "manifest.json").exists()
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest


def test_all_listed_files_exist(converted):
    out, manifest = converted
    for name, rel in manifest["files"].items():
        assert (out / rel).exists(), f"missing CSV for {name}"
        assert (out / (rel + ".mtd")).exists(), f"missing MTD for {name}"


def test_required_keys_per_block(converted):
    _, manifest = converted
    n = manifest["config"]["n_layer"]
    files = manifest["files"]
    assert "wte" in files and "wpe" in files
    assert "lnf_gamma" in files and "lnf_beta" in files
    expected = (
        "ln1_gamma ln1_beta "
        "W_Q W_K W_V b_Q b_K b_V "
        "W_context b_context "
        "ln2_gamma ln2_beta "
        "W_intermediate b_intermediate W_out b_out"
    ).split()
    for i in range(n):
        for suffix in expected:
            key = f"h{i}_{suffix}"
            assert key in files, f"missing manifest key {key}"


def test_qkv_split_shapes(converted):
    out, manifest = converted
    cfg = manifest["config"]
    D = cfg["n_embd"]
    for i in range(cfg["n_layer"]):
        for k in ("W_Q", "W_K", "W_V"):
            shape = manifest["shapes"][f"h{i}_{k}"]
            assert shape == [D, D], f"h{i}_{k}: got {shape}, want [{D},{D}]"
        for k in ("b_Q", "b_K", "b_V"):
            shape = manifest["shapes"][f"h{i}_{k}"]
            assert shape == [1, D], f"h{i}_{k}: got {shape}, want [1,{D}]"


def test_qkv_split_values_match_huggingface(converted):
    """Concatenating our split Q|K|V back together must equal HF's c_attn."""
    pytest.importorskip("transformers")
    from transformers import GPT2LMHeadModel

    out, manifest = converted
    cfg = manifest["config"]
    D = cfg["n_embd"]

    sd = GPT2LMHeadModel.from_pretrained(TINY).transformer.state_dict()

    for i in range(cfg["n_layer"]):
        Wc = sd[f"h.{i}.attn.c_attn.weight"].detach().cpu().numpy().astype(np.float64)
        bc = sd[f"h.{i}.attn.c_attn.bias"].detach().cpu().numpy().astype(np.float64)

        W_Q = _load_csv(out / manifest["files"][f"h{i}_W_Q"])
        W_K = _load_csv(out / manifest["files"][f"h{i}_W_K"])
        W_V = _load_csv(out / manifest["files"][f"h{i}_W_V"])
        b_Q = _load_csv(out / manifest["files"][f"h{i}_b_Q"]).reshape(-1)
        b_K = _load_csv(out / manifest["files"][f"h{i}_b_K"]).reshape(-1)
        b_V = _load_csv(out / manifest["files"][f"h{i}_b_V"]).reshape(-1)

        recon_W = np.concatenate([W_Q, W_K, W_V], axis=1)
        recon_b = np.concatenate([b_Q, b_K, b_V])

        np.testing.assert_allclose(recon_W, Wc, rtol=0, atol=0,
                                   err_msg=f"layer {i} W mismatch")
        np.testing.assert_allclose(recon_b, bc, rtol=0, atol=0,
                                   err_msg=f"layer {i} b mismatch")
        assert W_Q.shape == (D, D)


def test_mtd_metadata_well_formed(converted):
    out, manifest = converted
    for name, rel in manifest["files"].items():
        mtd = json.loads((out / (rel + ".mtd")).read_text())
        assert mtd["data_type"] == "matrix"
        assert mtd["value_type"] == "double"
        assert mtd["format"] == "csv"
        assert mtd["header"] is False
        assert mtd["rows"] == manifest["shapes"][name][0]
        assert mtd["cols"] == manifest["shapes"][name][1]


def test_no_lm_head_file(converted):
    """lm_head must be tied, not duplicated."""
    _, manifest = converted
    assert "lm_head" not in manifest["files"]
    assert manifest["tied"]["lm_head"] == "wte"
