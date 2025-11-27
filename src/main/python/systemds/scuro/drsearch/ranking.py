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

from dataclasses import replace
from typing import Callable, Iterable, List, Optional


def rank_by_tradeoff(
    entries: Iterable,
    *,
    weights = (0.7, 0.3),
    performance_metric_name: str = "accuracy",
    runtime_accessor: Optional[Callable[[object], float]] = None,
    cache_scores: bool = True,
    score_attr: str = "tradeoff_score",
) -> List:
    entries = list(entries)
    if not entries:
        return []

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

    performance = [float(performance_score_accessor(e)) for e in entries]
    runtimes = [float(runtime_accessor(e)) for e in entries]

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
            if hasattr(entry, score_attr):
                try:
                    new_entry = replace(entry, **{score_attr: score})
                    entries[entries.index(entry)] = new_entry
                except TypeError:
                    setattr(entry, score_attr, score)
            else:
                setattr(entry, score_attr, score)

    return sorted(
        entries, key=lambda entry: getattr(entry, score_attr, 0.0), reverse=True
    )
