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
"""Objectives and Pareto ranking shared by all optimization stages.

An objective is a ``(name, direction)`` pair with direction ``"max"`` or
``"min"``. Values are read from ``entry.val_score[name]``, so any metric of
the task and ``INFERENCE_TIME`` can be used, e.g.
``[("accuracy", "max"), (INFERENCE_TIME, "min")]``.
"""

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from systemds.scuro.drsearch.ranking import tie_broken
from systemds.scuro.drsearch.task import INFERENCE_TIME, PerformanceMeasure

Objective = Tuple[str, str]

# The value measured during the search, kept after a serial re-timing pass.
SEARCH_INFERENCE_TIME = "search_" + INFERENCE_TIME

__all__ = [
    "INFERENCE_TIME",
    "Objective",
    "crowding_distance",
    "dominates",
    "non_dominated_sort",
    "objective_vector",
    "pareto_front",
    "rank_by_pareto",
    "retime_front",
    "SEARCH_INFERENCE_TIME",
    "validate_objectives",
]

# Search-cost attributes some result types carry next to val_score.
_ATTRIBUTE_OBJECTIVES = ("runtime", "task_time", "representation_time")


def validate_objectives(objectives) -> List[Objective]:
    objectives = [tuple(objective) for objective in objectives]
    if not objectives:
        raise ValueError("objectives must contain at least one (name, direction) pair")
    for name, direction in objectives:
        if direction not in ("max", "min"):
            raise ValueError(
                f"objective direction must be 'max' or 'min', got "
                f"{direction!r} for objective {name!r}"
            )
    return objectives


def objective_vector(entry, objectives: Sequence[Objective]) -> Tuple[float, ...]:
    """Objective values of ``entry``; NaN where a value is missing."""
    scores = getattr(entry, "val_score", None)
    if isinstance(scores, PerformanceMeasure):
        scores = scores.average_scores
    values = []
    for name, _ in objectives:
        if isinstance(scores, dict) and name in scores:
            value = scores[name]
        elif name in _ATTRIBUTE_OBJECTIVES:
            value = getattr(entry, name, math.nan)
        else:
            value = math.nan
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(math.nan)
    return tuple(values)


def _oriented(vector, objectives):
    """Flip minimized objectives so that larger is better everywhere."""
    return [
        v if direction == "max" else -v for v, (_, direction) in zip(vector, objectives)
    ]


def dominates(a, b, objectives: Sequence[Objective]) -> bool:
    """True if objective vector ``a`` Pareto-dominates ``b``."""
    a, b = _oriented(a, objectives), _oriented(b, objectives)
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def non_dominated_sort(vectors, objectives: Sequence[Objective]) -> List[List[int]]:
    """Indices of ``vectors`` grouped into fronts, the Pareto front first."""
    n = len(vectors)
    dominated = [[] for _ in range(n)]
    dominator_count = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(vectors[i], vectors[j], objectives):
                dominated[i].append(j)
                dominator_count[j] += 1
            elif dominates(vectors[j], vectors[i], objectives):
                dominated[j].append(i)
                dominator_count[i] += 1

    fronts = []
    current = [i for i in range(n) if dominator_count[i] == 0]
    while current:
        fronts.append(current)
        following = []
        for i in current:
            for j in dominated[i]:
                dominator_count[j] -= 1
                if dominator_count[j] == 0:
                    following.append(j)
        current = sorted(following)
    return fronts


def crowding_distance(vectors, front: Sequence[int]) -> Dict[int, float]:
    """NSGA-II crowding distance of each index in ``front``; boundary points get inf."""
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    for m in range(len(vectors[front[0]])):
        ordered = sorted(front, key=lambda i: vectors[i][m])
        low, high = vectors[ordered[0]][m], vectors[ordered[-1]][m]
        distance[ordered[0]] = distance[ordered[-1]] = math.inf
        if high == low:
            continue
        for prev, cur, nxt in zip(ordered, ordered[1:], ordered[2:]):
            distance[cur] += (vectors[nxt][m] - vectors[prev][m]) / (high - low)
    return distance


def _valid(entries, objectives):
    """(index, entry, vector) for entries with a finite value for every objective."""
    rows = []
    for index, entry in enumerate(entries):
        if entry is None:
            continue
        vector = objective_vector(entry, objectives)
        if all(math.isfinite(v) for v in vector):
            rows.append((index, entry, vector))
    return rows


def _ranked(entries, objectives):
    """(front rank, original index, entry) in rank_by_pareto order."""
    objectives = validate_objectives(objectives)
    rows = _valid(list(entries), objectives)
    if not rows:
        return []
    indices, valid_entries, vectors = zip(*rows)
    tie_position = {local: pos for pos, local in enumerate(tie_broken(valid_entries))}

    keys = {}
    for front_rank, front in enumerate(non_dominated_sort(vectors, objectives)):
        crowding = (
            crowding_distance(vectors, front)
            if len(objectives) > 1
            else dict.fromkeys(front, 0.0)
        )
        for local in front:
            keys[local] = (front_rank, -crowding[local], tie_position[local])

    return [
        (keys[local][0], indices[local], valid_entries[local])
        for local in sorted(keys, key=keys.get)
    ]


def rank_by_pareto(entries, objectives: Sequence[Objective]):
    """Order entries by Pareto front, then by crowding distance (more isolated
    first), then by ``tie_broken``. Entries missing an objective are dropped.

    With a single objective this is a sort by that objective. Returns
    ``(sorted_entries, sorted_indices)`` like the other ``rank_by_*`` functions.
    """
    ranked = _ranked(entries, objectives)
    return [entry for _, _, entry in ranked], [index for _, index, _ in ranked]


def pareto_front(entries, objectives: Sequence[Objective]) -> list:
    """The non-dominated entries, in ``rank_by_pareto`` order."""
    return [entry for rank, _, entry in _ranked(entries, objectives) if rank == 0]


def _val_scores(entry) -> dict:
    scores = entry.val_score
    return scores.average_scores if isinstance(scores, PerformanceMeasure) else scores


def retime_front(
    entries,
    objectives: Sequence[Objective],
    measure: Callable[[object], float],
    repeats: int = 3,
) -> list:
    """Re-time the Pareto front serially and return the front after re-timing.

    Timings taken while other candidates run in parallel are too noisy to rank
    near-identical pipelines. ``measure(entry)`` re-measures the entry's
    ``INFERENCE_TIME`` alone; it is called ``repeats`` times per front member and
    the median replaces the search value, which is kept under
    ``SEARCH_INFERENCE_TIME``. Entries that join the front after re-timing are
    re-timed as well, until every front member has been measured serially.
    """
    if INFERENCE_TIME not in (name for name, _ in objectives):
        return pareto_front(entries, objectives)
    retimed = set()
    while True:
        front = pareto_front(entries, objectives)
        pending = [entry for entry in front if id(entry) not in retimed]
        if not pending:
            return front
        for entry in pending:
            scores = _val_scores(entry)
            scores.setdefault(SEARCH_INFERENCE_TIME, scores[INFERENCE_TIME])
            scores[INFERENCE_TIME] = float(
                np.median([measure(entry) for _ in range(max(1, repeats))])
            )
            retimed.add(id(entry))
