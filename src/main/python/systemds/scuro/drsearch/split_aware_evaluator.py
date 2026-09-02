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
import copy
import time
from dataclasses import dataclass

import numpy as np

from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    pushdown_aggregation_for_node,
)
from systemds.scuro.drsearch.task import PerformanceMeasure
from systemds.scuro.modality.indexed import IndexedModality
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.context import Context
from systemds.scuro.representations.dimensionality_reduction import (
    DimensionalityReduction,
)
from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.representation import (
    FitRequirement,
    Representation,
)
from systemds.scuro.representations.unimodal import UnimodalRepresentation


@dataclass
class SplitEvaluationResult:
    scores: list
    representation_time: float
    task_time: float


def dag_requires_fitting(dag: RepresentationDag) -> bool:
    return any(
        node.operation is not None
        and getattr(node.operation, "fit_requirement", FitRequirement.NONE)
        is not FitRequirement.NONE
        for node in dag.nodes
    )


class SplitAwareDAGEvaluator:
    def __init__(self, dag: RepresentationDag, modalities, task):
        self.dag = dag
        self.modalities = (
            list(modalities) if isinstance(modalities, (list, tuple)) else [modalities]
        )
        self.task = task

    @staticmethod
    def materialize(modality) -> None:
        loader = getattr(modality, "data_loader", None)
        if modality.has_data():
            if loader is None:
                return
            expected_instances = len(getattr(loader, "indices", ()))
            if not expected_instances or len(modality.data) == expected_instances:
                return

        if loader is None:
            raise ValueError("Split-aware evaluation requires materialized data")

        if getattr(loader, "chunk_size", None) is None:
            modality.extract_raw_data()
            return

        full_loader = copy.copy(loader)
        full_loader.chunk_size = None
        full_loader.reset()
        data, metadata = full_loader.load()
        modality._data = data
        modality.metadata = metadata

    @staticmethod
    def _wrap(transformed, instance_ids) -> IndexedModality:
        return IndexedModality(transformed, instance_ids)

    def _apply_stateless(self, operation, inputs, node):
        aggregation = pushdown_aggregation_for_node(node.parameters)

        if isinstance(operation, Fusion):
            instance_ids = operation._validate_indexed_modalities(inputs)
            transformed = inputs[0].modality.combine(
                [item.modality for item in inputs[1:]], operation
            )
            return self._wrap(transformed, instance_ids)
        elif isinstance(operation, Context):
            transformed = inputs.modality.context(operation)
        elif isinstance(operation, DimensionalityReduction):
            if getattr(operation, "needs_training", False):
                raise NotImplementedError(
                    "Trainable dimensionality reduction is not yet implemented for split-aware evaluation"
                )
            transformed = inputs.modality.dimensionality_reduction(operation)
        elif isinstance(operation, AggregatedRepresentation):
            transformed = operation.transform(inputs.modality)
        elif isinstance(operation, UnimodalRepresentation):
            transformed = inputs.modality.apply_representation(
                operation, aggregation=aggregation
            )
        elif isinstance(operation, Representation):
            transformed = inputs.modality.apply_representation(
                operation, aggregation=aggregation
            )
        else:
            raise TypeError(
                f"Unsupported unimodal operation: {type(operation).__name__}"
            )

        return self._wrap(transformed, inputs.instance_ids)

    @staticmethod
    def _leaf_input(partition, node):
        if len(partition) == 1:
            return partition[0]
        modality_id = node.modality_id
        try:
            modality_id = int(modality_id)
        except (TypeError, ValueError):
            pass
        matches = [
            item for item in partition if item.modality.modality_id == modality_id
        ]
        if not matches:
            raise KeyError(f"No modality found for leaf {node.node_id}")
        if node.representation_index in (None, -1) or len(matches) == 1:
            return matches[0]
        index = int(node.representation_index)
        if index >= len(matches):
            raise IndexError(f"No representation {index} for leaf {node.node_id}")
        return matches[index]

    def _execute_partitions(self, *partitions):
        cache = {}

        def execute_node(node_id):
            if node_id in cache:
                return cache[node_id]

            node = self.dag.get_node_by_id(node_id)
            if not node.inputs:
                result = tuple(
                    self._leaf_input(partition, node) for partition in partitions
                )
                cache[node_id] = result
                return result

            input_results = [execute_node(input_id) for input_id in node.inputs]
            input_partitions = tuple(
                tuple(result[i] for result in input_results)
                for i in range(len(partitions))
            )
            input_partitions = tuple(
                items[0] if len(items) == 1 else items for items in input_partitions
            )
            operation = node.operation(params=node.parameters)
            requirement = getattr(operation, "fit_requirement", FitRequirement.NONE)

            if requirement is not FitRequirement.NONE:
                labels = None
                fit_input = input_partitions[0]
                fit_ids = (
                    fit_input[0].instance_ids
                    if isinstance(fit_input, tuple)
                    else fit_input.instance_ids
                )
                if requirement is FitRequirement.SUPERVISED:
                    labels = np.asarray(self.task.labels)[list(fit_ids)]
                fit_state = operation.fit(fit_input, labels=labels)
                aggregation = pushdown_aggregation_for_node(node.parameters)
                result = tuple(
                    operation.transform_fitted(
                        partition,
                        fit_state,
                        aggregation=aggregation,
                    )
                    for partition in input_partitions
                )
            else:
                result = tuple(
                    self._apply_stateless(operation, partition, node)
                    for partition in input_partitions
                )

            cache[node_id] = result
            return result

        return execute_node(self.dag.root_node_id)

    def _indexed_modalities(self):
        full = []
        for modality in self.modalities:
            self.materialize(modality)
            full.append(IndexedModality(modality, range(len(modality.data))))
        return tuple(full)

    def fit_transform(self, fit_indices, transform_indices) -> IndexedModality:
        full = self._indexed_modalities()
        fit_partition = tuple(item.subset(fit_indices) for item in full)
        transform_partition = tuple(item.subset(transform_indices) for item in full)
        _, transformed = self._execute_partitions(
            fit_partition,
            transform_partition,
        )
        return transformed

    @staticmethod
    def _partition_data(partition):
        return list(partition.modality.data)

    def evaluate(self) -> SplitEvaluationResult:
        full = self._indexed_modalities()

        train_scores = PerformanceMeasure("train", self.task.performance_measures)
        val_scores = PerformanceMeasure("val", self.task.performance_measures)
        test_scores = PerformanceMeasure("test", self.task.performance_measures)
        self.task.training_time = []
        self.task.inference_time = []

        representation_time = 0.0
        task_time = 0.0
        for split in self.task.evaluation_splits:
            train = tuple(item.subset(split.fit_indices) for item in full)
            evaluation = tuple(item.subset(split.evaluation_indices) for item in full)
            test = tuple(item.subset(self.task.test_indices) for item in full)

            start = time.perf_counter()
            transformed_train, transformed_evaluation, transformed_test = (
                self._execute_partitions(train, evaluation, test)
            )
            representation_time += time.perf_counter() - start

            model = self.task.create_model()
            train_X = self._partition_data(transformed_train)
            evaluation_X = self._partition_data(transformed_evaluation)
            test_X = self._partition_data(transformed_test)
            train_y = np.asarray(self.task.labels)[list(split.fit_indices)]
            evaluation_y = np.asarray(self.task.labels)[list(split.evaluation_indices)]
            test_y = np.asarray(self.task.labels)[list(self.task.test_indices)]

            start = time.perf_counter()
            train_start = time.perf_counter()
            train_score = model.fit(
                train_X,
                train_y,
                evaluation_X,
                evaluation_y,
            )
            self.task.training_time.append(time.perf_counter() - train_start)
            val_score = model.test(evaluation_X, evaluation_y)
            test_start = time.perf_counter()
            test_score = model.test(np.asarray(test_X), test_y)
            self.task.inference_time.append(time.perf_counter() - test_start)
            task_time += time.perf_counter() - start

            train_scores.add_scores(train_score[0])
            val_scores.add_scores(val_score[0])
            test_scores.add_scores(test_score[0])

            if hasattr(model, "clean_up"):
                model.clean_up()

        self.task.last_run_timing = {
            "train_time_per_fold_s": list(self.task.training_time),
            "test_inference_time_per_fold_s": list(self.task.inference_time),
            "train_time_mean_s": (
                float(np.mean(self.task.training_time))
                if self.task.training_time
                else 0.0
            ),
            "test_inference_time_mean_s": (
                float(np.mean(self.task.inference_time))
                if self.task.inference_time
                else 0.0
            ),
            "n_test_instances": len(self.task.test_indices),
        }

        return SplitEvaluationResult(
            scores=[
                train_scores.compute_averages(),
                val_scores.compute_averages(),
                test_scores.compute_averages(),
            ],
            representation_time=representation_time,
            task_time=task_time,
        )
