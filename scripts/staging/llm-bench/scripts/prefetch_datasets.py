#!/usr/bin/env python3
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

"""
Prefetch HuggingFace datasets used by default workload configs.

Run from the llm-bench directory after `pip install -r requirements.txt`:

    python scripts/prefetch_datasets.py
    python scripts/prefetch_datasets.py --extra

Requires network access on first run; data is cached under ~/.cache/huggingface
(or HF_HOME / HF_DATASETS_CACHE if set).
"""

from __future__ import annotations

import argparse
import sys


def _prefetch(name: str, fn) -> None:
    print(f"Prefetching: {name} ...", flush=True)
    fn()
    print(f"  OK: {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch HuggingFace datasets for llm-bench")
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Also load optional dataset sources (CNN/DailyMail, LogiQA) used when "
        "workload configs select them.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: install dependencies: pip install -r requirements.txt", file=sys.stderr)
        return 1

    def gsm8k() -> None:
        load_dataset("openai/gsm8k", "main", split="test")

    def boolq() -> None:
        load_dataset("google/boolq", split="validation")

    def xsum() -> None:
        load_dataset("EdinburghNLP/xsum", split="test")

    def conll2003() -> None:
        try:
            load_dataset("conll2003", split="test")
        except Exception:
            load_dataset("eriktks/conll2003", split="test")

    def stsb() -> None:
        load_dataset("mteb/stsbenchmark-sts", split="test")

    for label, fn in (
        ("openai/gsm8k (math)", gsm8k),
        ("google/boolq (reasoning)", boolq),
        ("EdinburghNLP/xsum (summarization)", xsum),
        ("CoNLL-2003 (json_extraction)", conll2003),
        ("mteb/stsbenchmark-sts (embeddings)", stsb),
    ):
        _prefetch(label, fn)

    if args.extra:

        def cnn() -> None:
            load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")

        def logiqa() -> None:
            load_dataset("lucasmccabe/logiqa", split="test")

        for label, fn in (
            ("abisee/cnn_dailymail (summarization alt.)", cnn),
            ("lucasmccabe/logiqa (reasoning alt.)", logiqa),
        ):
            _prefetch(label, fn)

    print("Done. Datasets are cached for offline use by the workload loaders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
