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
# Unless required by applicable law or agreed in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
from typing import List, Any
from functools import lru_cache

from systemds.scuro import ModalityType
from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.hadamard import Hadamard
from systemds.scuro.representations.sum import Sum
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.utils.schema_helpers import get_shape
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationNode,
    RepresentationDAGBuilder,
)
from systemds.scuro.drsearch.representation_dag_visualizer import visualize_dag


class UnimodalOptimizer:
    def __init__(self, modalities, tasks, debug=True):
        self.modalities = modalities
        self.tasks = tasks
        self.run = None

        self.builders = {
            modality.modality_id: RepresentationDAGBuilder() for modality in modalities
        }

        self.debug = debug

        self.operator_registry = Registry()
        self.operator_performance = UnimodalResults(modalities, tasks, debug, self.run)

        self._tasks_require_same_dims = True
        self.expected_dimensions = tasks[0].expected_dim

        for i in range(1, len(tasks)):
            self.expected_dimensions = tasks[i].expected_dim
            if tasks[i - 1].expected_dim != tasks[i].expected_dim:
                self._tasks_require_same_dims = False

        self._combination_operators = [Concatenation(), Hadamard(), Sum()]

    @lru_cache(maxsize=128)
    def _get_modality_operators(self, modality_type):
        return self.operator_registry.get_representations(modality_type)

    @lru_cache(maxsize=128)
    def _get_not_self_contained_reps(self, modality_type):
        return self.operator_registry.get_not_self_contained_representations(
            modality_type
        )

    @lru_cache(maxsize=32)
    def _get_context_operators(self):
        return self.operator_registry.get_context_operators()

    def store_results(self, file_name=None):
        if file_name is None:
            import time

            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = "unimodal_optimizer" + timestr + ".pkl"

        with open(file_name, "wb") as f:
            pickle.dump(self.operator_performance.results, f)

    def load_results(self, file_name):
        with open(file_name, "rb") as f:
            self.operator_performance.results = pickle.load(f)
            self.operator_performance.cache = None

    def optimize_parallel(self, n_workers=None):
        if n_workers is None:
            n_workers = min(len(self.modalities), mp.cpu_count())

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_modality = {
                executor.submit(self._process_modality, modality, True): modality
                for modality in self.modalities
            }

            for future in as_completed(future_to_modality):
                modality = future_to_modality[future]
                results = future.result()
                self._merge_results(results)

    def optimize(self):
        """Optimize representations for each modality"""
        for modality in self.modalities:
            local_result = self._process_modality(modality, False)

    def _process_modality(self, modality, parallel):
        if parallel:
            local_results = UnimodalResults([modality], self.tasks, debug=False)
        else:
            local_results = self.operator_performance

        modality_specific_operators = self._get_modality_operators(
            modality.modality_type
        )

        for operator in modality_specific_operators:
            dags = self._build_modality_dag(modality, operator())

            for dag in dags:
                representations = dag.execute([modality])
                node_id = list(representations.keys())[-1]
                node = dag.get_node_by_id(node_id)
                if node.operation is None:
                    continue

                reps = self._get_representation_chain(node, dag)
                combination = next((op for op in reps if isinstance(op, Fusion)), None)
                self._evaluate_local(
                    representations[node_id], local_results, dag, combination
                )
                if self.debug:
                    visualize_dag(dag)

        return local_results

    def _get_representation_chain(
        self, node: "RepresentationNode", dag: RepresentationDag
    ) -> List[Any]:
        representations = []
        if node.operation:
            representations.append(node.operation)

        for input_id in node.inputs:
            input_node = dag.get_node_by_id(input_id)
            if input_node.operation:
                representations.extend(self._get_representation_chain(input_node, dag))

        return representations

    def _merge_results(self, local_results):
        for modality_id in local_results.results:
            for task_name in local_results.results[modality_id]:
                self.operator_performance.results[modality_id][task_name].extend(
                    local_results.results[modality_id][task_name]
                )

        for modality in self.modalities:
            for task_name in local_results.cache[modality]:
                for key, value in local_results.cache[modality][task_name].items():
                    self.operator_performance.cache[modality][task_name][key] = value

    def _evaluate_local(self, modality, local_results, dag, combination=None):
        if self._tasks_require_same_dims:
            if self.expected_dimensions == 1 and get_shape(modality.metadata) > 1:
                builder = self.builders[modality.modality_id]
                agg_operator = AggregatedRepresentation()
                rep_node_id = builder.create_operation_node(
                    agg_operator.__class__, [dag.root_node_id], agg_operator.parameters
                )
                dag = builder.build(rep_node_id)
                representations = dag.execute([modality])
                node_id = list(representations.keys())[-1]
                for task in self.tasks:
                    start = time.perf_counter()
                    scores = task.run(representations[node_id].data)
                    end = time.perf_counter()

                    local_results.add_result(
                        scores, modality, task.model.name, end - start, combination, dag
                    )
            else:
                modality.pad()
                for task in self.tasks:
                    start = time.perf_counter()
                    scores = task.run(modality.data)
                    end = time.perf_counter()
                    local_results.add_result(
                        scores, modality, task.model.name, end - start, combination, dag
                    )
        else:
            for task in self.tasks:
                if task.expected_dim == 1 and get_shape(modality.metadata) > 1:
                    builder = self.builders[modality.modality_id]
                    agg_operator = AggregatedRepresentation(Aggregation())
                    rep_node_id = builder.create_operation_node(
                        agg_operator.__class__,
                        [dag.root_node_id],
                        agg_operator.parameters,
                    )
                    dag = builder.build(rep_node_id)
                    representations = dag.execute([modality])
                    node_id = list(representations.keys())[-1]

                    start = time.perf_counter()
                    scores = task.run(representations[node_id].data)
                    end = time.perf_counter()
                    local_results.add_result(
                        scores, modality, task.model.name, end - start, combination, dag
                    )
                else:
                    start = time.perf_counter()
                    scores = task.run(modality.data)
                    end = time.perf_counter()
                    local_results.add_result(
                        scores, modality, task.model.name, end - start, combination, dag
                    )

    def _build_modality_dag(
        self, modality: Modality, operator: Any
    ) -> List[RepresentationDag]:
        dags = []
        builder = self.builders[modality.modality_id]
        leaf_id = builder.create_leaf_node(modality.modality_id)

        rep_node_id = builder.create_operation_node(
            operator.__class__, [leaf_id], operator.parameters
        )
        current_node_id = rep_node_id
        dags.append(builder.build(current_node_id))

        if not operator.self_contained:
            not_self_contained_reps = self._get_not_self_contained_reps(
                modality.modality_type
            )
            not_self_contained_reps = [
                rep for rep in not_self_contained_reps if rep != operator.__class__
            ]

            for combination in self._combination_operators:
                current_node_id = rep_node_id
                for other_rep in not_self_contained_reps:
                    other_rep_id = builder.create_operation_node(
                        other_rep, [leaf_id], other_rep().parameters
                    )

                    combine_id = builder.create_operation_node(
                        combination.__class__,
                        [current_node_id, other_rep_id],
                        combination.parameters,
                    )
                    dags.append(builder.build(combine_id))
                    current_node_id = combine_id
            if modality.modality_type in [
                ModalityType.EMBEDDING,
                ModalityType.IMAGE,
                ModalityType.AUDIO,
            ]:
                dags.extend(
                    self.default_context_operators(
                        modality, builder, leaf_id, current_node_id
                    )
                )
            elif modality.modality_type == ModalityType.TIMESERIES:
                dags.extend(
                    self.temporal_context_operators(
                        modality, builder, leaf_id, current_node_id
                    )
                )
        return dags

    def default_context_operators(self, modality, builder, leaf_id, current_node_id):
        dags = []
        context_operators = self._get_context_operators()
        for context_op in context_operators:
            if (
                modality.modality_type != ModalityType.TEXT
                and modality.modality_type != ModalityType.VIDEO
            ):
                context_node_id = builder.create_operation_node(
                    context_op,
                    [leaf_id],
                    context_op().parameters,
                )
                dags.append(builder.build(context_node_id))

            context_node_id = builder.create_operation_node(
                context_op,
                [current_node_id],
                context_op().parameters,
            )
            dags.append(builder.build(context_node_id))

        return dags

    def temporal_context_operators(self, modality, builder, leaf_id, current_node_id):
        aggregators = self.operator_registry.get_representations(modality.modality_type)
        context_operators = self._get_context_operators()

        dags = []
        for agg in aggregators:
            for context_operator in context_operators:
                context_node_id = builder.create_operation_node(
                    context_operator,
                    [leaf_id],
                    context_operator(agg()).parameters,
                )
                dags.append(builder.build(context_node_id))

        return dags


class UnimodalResults:
    def __init__(self, modalities, tasks, debug=False, run=None):
        self.modality_ids = [modality.modality_id for modality in modalities]
        self.task_names = [task.model.name for task in tasks]
        self.results = {}
        self.debug = debug
        self.cache = {}

        for modality in self.modality_ids:
            self.results[modality] = {task_name: [] for task_name in self.task_names}
            self.cache[modality] = {task_name: {} for task_name in self.task_names}

    def add_result(self, scores, modality, task_name, task_time, combination, dag):
        entry = ResultEntry(
            train_score=scores[0],
            val_score=scores[1],
            representation_time=modality.transform_time,
            task_time=task_time,
            combination=combination.name if combination else "",
            dag=dag,
        )

        self.results[modality.modality_id][task_name].append(entry)

        cache_key = (
            id(dag),
            scores[1],
            modality.transform_time,
        )
        self.cache[modality.modality_id][task_name][cache_key] = modality

        if self.debug:
            print(f"{modality.modality_id}_{task_name}: {entry}")

    def print_results(self):
        for modality in self.modality_ids:
            for task_name in self.task_names:
                for entry in self.results[modality][task_name]:
                    print(f"{modality}_{task_name}: {entry}")

    def get_k_best_results(self, modality, k, task):
        """
        Get the k best results for the given modality
        :param modality: modality to get the best results for
        :param k: number of best results
        """
        task_results = self.results[modality.modality_id][task.model.name]

        results = sorted(task_results, key=lambda x: x.val_score, reverse=True)[:k]

        sorted_indices = sorted(
            range(len(task_results)),
            key=lambda x: task_results[x].val_score,
            reverse=True,
        )[:k]
        if not self.cache:
            cache = [
                list(task_results[i].dag.execute([modality]).values())[-1]
                for i in sorted_indices
            ]
        else:
            cache_items = (
                list(self.cache[modality.modality_id][task.model.name].items())
                if self.cache[modality.modality_id][task.model.name]
                else []
            )
            cache = [cache_items[i][1] for i in sorted_indices if i < len(cache_items)]

        return results, cache


@dataclass(frozen=True)
class ResultEntry:
    val_score: float
    train_score: float
    representation_time: float
    task_time: float
    combination: str
    dag: RepresentationDag
