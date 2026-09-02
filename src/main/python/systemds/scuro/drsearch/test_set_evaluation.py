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
from __future__ import annotations

import time
from typing import Any, Dict, List

from systemds.scuro.modality.modality import Modality
from systemds.scuro.drsearch.representation_dag import RepresentationDag
from systemds.scuro.drsearch.split_aware_evaluator import (
    SplitAwareDAGEvaluator,
    dag_requires_fitting,
)
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.utils.schema_helpers import get_shape


def _unwrap(result):
    if isinstance(result, dict):
        if not result:
            return None
        return result[list(result.keys())[-1]]
    return result


def _match_expected_dim(modality, task):
    if modality is None or task is None:
        return modality
    if getattr(task, "expected_dim", 1) == 1 and get_shape(modality.metadata) > 1:
        return AggregatedRepresentation().transform(modality)
    return modality


def measure_representation_time_on_test_set(
    dag: RepresentationDag,
    modalities: List[Modality],
    test_indices: List[int],
    task=None,
    repeats: int = 1,
) -> Dict[str, Any]:
    subsets = [modality.subset(test_indices) for modality in modalities]

    timings = []
    output = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        if dag_requires_fitting(dag):
            if task is None:
                raise ValueError("A task is required to fit the representation DAG")
            transformed = SplitAwareDAGEvaluator(dag, modalities, task).fit_transform(
                task.train_indices, test_indices
            )
            output = _match_expected_dim(transformed.modality, task)
        else:
            output = _match_expected_dim(
                _unwrap(dag.execute(subsets, task, enable_cache=False)), task
            )
        timings.append(time.perf_counter() - t0)

    timings.sort()
    median = timings[len(timings) // 2]
    n_test = len(test_indices)

    output_shape = None
    if output is not None and getattr(output, "data", None) is not None:
        try:
            output_shape = tuple(getattr(output.data[0], "shape", ()))
        except (IndexError, TypeError):
            output_shape = None

    return {
        "test_only_representation_time_s": median,
        "test_only_representation_time_all_runs_s": timings,
        "test_only_representation_time_per_instance_ms": (
            median / n_test * 1000.0 if n_test else 0.0
        ),
        "test_only_n_instances": n_test,
        "test_only_output_shape": output_shape,
        "test_only_features_are_valid_for_scoring": False,
    }


def measure_test_set_application(
    dag: RepresentationDag,
    modalities: List[Modality],
    task,
    full_data=None,
    repeats: int = 1,
    latency_repeats: int = 200,
    latency_warmup: int = 20,
) -> Dict[str, Any]:
    record = measure_representation_time_on_test_set(
        dag, modalities, task.test_indices, task=task, repeats=repeats
    )

    if full_data is None:
        t0 = time.perf_counter()
        if dag_requires_fitting(dag):
            transformed = SplitAwareDAGEvaluator(dag, modalities, task).fit_transform(
                task.train_indices, range(len(modalities[0].data))
            )
            output = _match_expected_dim(transformed.modality, task)
        else:
            output = _match_expected_dim(
                _unwrap(dag.execute(modalities, task, enable_cache=False)), task
            )
        record["full_representation_time_s"] = time.perf_counter() - t0
        full_data = None if output is None else output.data

    if full_data is not None:
        record.update(
            task.fit_once_and_time_inference(
                full_data,
                latency_repeats=latency_repeats,
                latency_warmup=latency_warmup,
            )
        )

    return record
