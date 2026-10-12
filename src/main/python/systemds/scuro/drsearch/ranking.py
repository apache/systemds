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

from typing import Callable, Iterable, Optional

import numpy as np


def _dag_size(entry) -> int:
    try:
        return sum(
            1
            for node in entry.dag.nodes
            if getattr(node, "operation", None) is not None
        )
    except Exception:
        return 0


def rank_by_robustness(
    entries: Iterable,
    *,
    performance_metric_name: str = "accuracy",
    standard_deviation_weight: float = 1.0,
    one_se_parsimony: bool = True,
    cache_scores: bool = True,
    score_attr: str = "robustness_score",
):
    entries = list(entries)
    if not entries:
        return [], []

    if not np.isfinite(standard_deviation_weight) or standard_deviation_weight < 0:
        raise ValueError("standard_deviation_weight must be finite and non-negative")

    def perf_of(entry):
        if entry is None:
            return None
        try:
            score = float(entry.val_score[performance_metric_name])
        except (KeyError, TypeError, ValueError):
            return None
        return score if np.isfinite(score) else None

    indexed_entries = [
        (index, entry, score)
        for index, entry in enumerate(entries)
        if (score := perf_of(entry)) is not None
    ]
    if not indexed_entries:
        return [], []

    def variation_of(entry):
        try:
            fold_values = np.asarray(
                entry.val_fold_scores[performance_metric_name], dtype=float
            )
            fold_values = fold_values[np.isfinite(fold_values)]
        except (AttributeError, KeyError, TypeError, ValueError):
            fold_values = np.array([], dtype=float)

        if fold_values.size > 1:
            return float(fold_values.std(ddof=1)), int(fold_values.size)

        try:
            std = float(entry.val_score_stds[performance_metric_name])
        except (AttributeError, KeyError, TypeError, ValueError):
            std = 0.0
        if not np.isfinite(std) or std < 0:
            std = 0.0
        return std, max(1, int(fold_values.size))

    original_indices, entries, performance = zip(*indexed_entries)
    entries = list(entries)
    perf = np.array(performance, dtype=float)
    sizes = np.array([_dag_size(e) if e is not None else 0 for e in entries], float)
    variations = [variation_of(entry) for entry in entries]
    stds = np.array([variation[0] for variation in variations], dtype=float)
    fold_counts = np.array([variation[1] for variation in variations], dtype=int)
    robustness = perf - standard_deviation_weight * stds

    if cache_scores:
        for entry, score in zip(entries, robustness):
            if entry is not None:
                setattr(entry, score_attr, float(score))

    if one_se_parsimony and len(entries) > 1:
        best_index = int(np.argmax(robustness))
        standard_error = stds[best_index] / np.sqrt(fold_counts[best_index])
        threshold = float(robustness[best_index] - standard_error)
        keys = [
            (True, -sz, float(s)) if s >= threshold else (False, 0.0, float(s))
            for s, sz in zip(robustness, sizes)
        ]
    else:
        keys = [(True, 0.0, float(s)) for s in robustness]

    local_indices = sorted(range(len(entries)), key=lambda i: keys[i], reverse=True)
    sorted_entries = [entries[i] for i in local_indices]
    sorted_indices = [original_indices[i] for i in local_indices]

    return sorted_entries, sorted_indices


def rank_by_tradeoff(
    entries: Iterable,
    *,
    weights=(1.0, 0.0),
    performance_metric_name: str = "accuracy",
    runtime_accessor: Optional[Callable[[object], float]] = None,
    cache_scores: bool = True,
    score_attr: str = "tradeoff_score",
):
    entries = list(entries)
    if not entries:
        return [], []

    performance_score_accessor = lambda entry: getattr(entry, "val_score")[
        performance_metric_name
    ]

    if runtime_accessor is None:

        def runtime_accessor(entry):
            if hasattr(entry, "runtime"):
                return getattr(entry, "runtime")
            rep = getattr(entry, "representation_time", 0.0)
            task = getattr(entry, "task_time", 0.0)
            return rep + task

    performance = [
        float(performance_score_accessor(e)) if e is not None else 0.0 for e in entries
    ]
    runtimes = [float(runtime_accessor(e)) if e is not None else 0.0 for e in entries]

    perf_min, perf_max = min(performance), max(performance)
    run_min, run_max = min(runtimes), max(runtimes)

    def safe_normalize(values, vmin, vmax):
        if vmax - vmin == 0.0:
            return [1.0] * len(values)
        return [(v - vmin) / (vmax - vmin) for v in values]

    norm_perf = safe_normalize(performance, perf_min, perf_max)
    norm_run = safe_normalize(runtimes, run_min, run_max)
    norm_run = [1.0 - r for r in norm_run]

    acc_w, run_w = weights
    total_w = (acc_w or 0.0) + (run_w or 0.0)
    if total_w == 0.0:
        acc_w = 1.0
        run_w = 0.0
    else:
        acc_w /= total_w
        run_w /= total_w

    scores = [acc_w * a + run_w * r for a, r in zip(norm_perf, norm_run)]

    if cache_scores:
        for entry, score in zip(entries, scores):
            if entry is None:
                continue
            if hasattr(entry, score_attr):
                setattr(entry, score_attr, score)
            else:
                setattr(entry, score_attr, score)

    sorted_entries = sorted(
        entries,
        key=lambda e: e.tradeoff_score if hasattr(e, "tradeoff_score") else 0.0,
        reverse=True,
    )

    sorted_indices = [
        i
        for i, _ in sorted(
            enumerate(entries),
            key=lambda pair: pair[1].tradeoff_score if pair is not None else None,
            reverse=True,
        )
    ]

    return sorted_entries, sorted_indices
