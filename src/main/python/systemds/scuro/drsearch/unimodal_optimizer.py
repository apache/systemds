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
import copy
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
from typing import List, Any, Optional, Dict
from functools import lru_cache

from systemds.scuro import ModalityType
from systemds.scuro.drsearch.node_executor import NodeExecutor, ResultEntry
from systemds.scuro.drsearch.ranking import rank_by_tradeoff
from systemds.scuro.drsearch.task import PerformanceMeasure
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.hadamard import Hadamard
from systemds.scuro.representations.sum import Sum
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.modality.modality import Modality
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.utils.checkpointing import CheckpointManager
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationNode,
    CSEAwareDAGBuilder,
    pushdown_aggregation,
)
from bisect import bisect_left


class UnimodalOptimizer:
    def __init__(
        self,
        modalities,
        tasks,
        debug=True,
        save_all_results=False,
        result_path=None,
        k=2,
        metric_name="accuracy",
        checkpoint_every: Optional[int] = 1,
        resume: bool = False,
        max_num_workers: int = -1,
    ):
        self.modalities = modalities
        self.tasks = tasks
        self.modality_ids = [modality.modality_id for modality in modalities]
        self.save_all_results = save_all_results
        self.result_path = result_path
        self.k = k
        self.metric_name = metric_name
        self.checkpoint_every = checkpoint_every
        self.resume = resume
        self.max_num_workers = max_num_workers
        self._checkpoint_manager = CheckpointManager(
            self.result_path or ".",
            "unimodal_checkpoint_",
            checkpoint_every=self.checkpoint_every,
            resume=self.resume,
        )
        # TODO: check if we should make this a local variable (might keep unnecessary memory usage)
        self.builders = {
            modality.modality_id: CSEAwareDAGBuilder() for modality in modalities
        }

        self.debug = debug

        self.operator_registry = Registry()
        self.operator_performance = UnimodalResults(
            modalities, tasks, debug, True, k, self.metric_name
        )

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
    def _get_context_operators(self, modality_type):
        return self.operator_registry.get_context_operators(modality_type)

    @lru_cache(maxsize=32)
    def _get_dimensionality_reduction_operators(self, modality_type):
        return self.operator_registry.get_dimensionality_reduction_operators(
            modality_type
        )

    def store_results(self, file_name=None):
        if file_name is None:
            import time

            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = "unimodal_optimizer" + timestr + ".pkl"

        file_name = f"{self.result_path}/{file_name}"
        with open(file_name, "wb") as f:
            pickle.dump(self.operator_performance.results, f)

    def store_cache(self, file_name=None):
        if file_name is None:
            import time

            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = "unimodal_optimizer_cache" + timestr + ".pkl"

        file_name = f"{self.result_path}/{file_name}"
        with open(file_name, "wb") as f:
            pickle.dump(self.operator_performance.cache, f)

    def load_results(self, file_name):
        with open(file_name, "rb") as f:
            self.operator_performance.results = pickle.load(f)

    def load_cache(self):
        for modality in self.modalities:
            for task in self.tasks:
                self.operator_performance.cache[modality.modality_id][
                    task.model.name
                ] = []
                with open(
                    f"{modality.modality_id}_{task.model.name}_cache.pkl", "rb"
                ) as f:
                    cache = pickle.load(f)
                    for c in cache:
                        self.operator_performance.cache[modality.modality_id][
                            task.model.name
                        ].append(c)

    def _count_results(self, results) -> int:
        count = 0
        for modality_id in results:
            for task_name in results[modality_id]:
                count += len(results[modality_id][task_name])
        return count

    def _count_results_by_modality(self, results) -> Dict[Any, int]:
        counts = {}
        for modality_id in results:
            counts[modality_id] = len(
                results[modality_id][list(results[modality_id].keys())[0]]
            )

        return counts

    def resume_from_checkpoint(self):
        loaded = self._checkpoint_manager.resume_from_checkpoint(
            "eval_count_by_modality", self._count_results_by_modality
        )
        if loaded:
            results, _, _ = loaded
            self.operator_performance.results = results

    def optimize_parallel(self, n_workers=None):
        if self.resume:
            self.resume_from_checkpoint()
            # TODO: check which modalities have been processed and skip the ones that have been processed

        if n_workers is None:
            n_workers = min(len(self.modalities), mp.cpu_count())

        with mp.Manager() as manager:

            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                future_to_modality = {
                    executor.submit(
                        self._process_modality,
                        modality,
                        self._checkpoint_manager.skip_remaining_by_key.get(
                            modality.modality_id, 0
                        )
                        / len(self.tasks),
                        scheduler=None,
                    ): modality
                    for modality in self.modalities
                }

                for future in as_completed(future_to_modality):
                    modality = future_to_modality[future]
                    try:
                        results = future.result()
                        self._merge_results(results)
                        new_count = self._count_results(results.results)
                        self._checkpoint_manager.increment(
                            modality.modality_id, new_count
                        )
                        self._checkpoint_manager.checkpoint_if_due(
                            self.operator_performance.results,
                        )
                    except Exception as e:
                        print(f"Error processing modality {modality.modality_id}: {e}")
                        import traceback

                        traceback.print_exc()
                        self._checkpoint_manager.save_checkpoint(
                            self.operator_performance.results,
                            {},
                        )
                        continue

    def optimize(self):
        if self.resume:
            self.resume_from_checkpoint()

        for modality in self.modalities:
            try:
                local_result = self._process_modality(
                    modality,
                    (
                        int(
                            round(
                                self._checkpoint_manager.skip_remaining_by_key.get(
                                    modality.modality_id, 0
                                )
                                / len(self.tasks)
                            )
                        )
                        if self.resume
                        else 0
                    ),
                )
                self._merge_results(local_result)
                new_count = self._count_results(local_result.results)
                self._checkpoint_manager.increment(modality.modality_id, new_count)
                self._checkpoint_manager.checkpoint_if_due(
                    self.operator_performance.results
                )
                if self.save_all_results:
                    self.store_results(f"{modality.modality_id}_unimodal_results.pkl")
            except Exception as e:
                print(f"Error processing modality {modality.modality_id}: {e}")
                import traceback

                traceback.print_exc()
                self._checkpoint_manager.save_checkpoint(
                    self.operator_performance.results, {}
                )
                raise

    def _expand_dags_with_task_roots(
        self, dags: List[RepresentationDag]
    ) -> List[RepresentationDag]:
        expanded_dags: List[RepresentationDag] = []

        for dag in dags:
            root_id = dag.root_node_id
            for task_idx, _ in enumerate(self.tasks):
                task_node_id = f"task_{root_id}_{task_idx}"

                task_node = RepresentationNode(
                    node_id=task_node_id,
                    operation=None,
                    inputs=[root_id],
                    parameters={
                        "_node_kind": "task",
                        "_task_idx": task_idx,
                        "_dag_root_id": root_id,
                    },
                )

                task_root_dag = RepresentationDag(
                    nodes=[*dag.nodes, task_node], root_node_id=task_node_id
                )
                expanded_dags.append(task_root_dag)
        return expanded_dags

    def _process_modality(self, modality, skip_remaining: int = 0, scheduler=None):
        local_results = UnimodalResults(
            [modality],
            self.tasks,
            debug=False,
            store_cache=False,
            metric_name=self.metric_name,
        )

        modality_specific_operators = self._get_modality_operators(
            modality.modality_type
        )
        dags = []
        for operator in modality_specific_operators:
            dags.extend(self._build_modality_dag(modality, operator()))

        if (
            modality.modality_type == ModalityType.TIMESERIES
            or modality.modality_type == ModalityType.AUDIO
        ):
            dags.extend(
                self.temporal_context_operators(
                    modality,
                    self.builders[modality.modality_id],
                    dags[0].get_leaf_node_id(),
                )
            )

        dags = self.add_aggregation_operator(self.builders[modality.modality_id], dags)
        dags = pushdown_aggregation(dags)

        if skip_remaining > 0:
            dags = dags[skip_remaining:]

        expanded_dags = self._expand_dags_with_task_roots(dags)

        node_executor = NodeExecutor(
            expanded_dags,
            [modality],
            self.tasks,
            self._checkpoint_manager,
            self.max_num_workers,
            self.result_path,
        )
        task_results = node_executor.run()

        for task_result in task_results:
            local_results.add_task_result(task_result)

        if self.save_all_results:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"{modality.modality_id}_unimodal_results_{timestr}.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(local_results.results, f)

        return local_results

    def _merge_results(self, local_results):
        for modality_id in local_results.results:
            for task_name in local_results.results[modality_id]:
                self.operator_performance.results[modality_id][task_name].extend(
                    local_results.results[modality_id][task_name]
                )

    def add_dimensionality_reduction_operators(self, builder, current_node_id):
        dags = []
        modality_type = (
            builder.get_node(current_node_id).operation().output_modality_type
        )

        if modality_type is not ModalityType.EMBEDDING:
            return None

        dimensionality_reduction_operators = (
            self._get_dimensionality_reduction_operators(modality_type)
        )
        for dimensionality_reduction_op in dimensionality_reduction_operators:
            dimensionality_reduction_node_id = builder.create_operation_node(
                dimensionality_reduction_op,
                [current_node_id],
                dimensionality_reduction_op().get_current_parameters(),
            )
            dags.append(builder.build(dimensionality_reduction_node_id))
        return dags

    def _build_modality_dag(
        self, modality: Modality, operator: Any
    ) -> List[RepresentationDag]:
        dags = []
        builder = self.builders[modality.modality_id]
        leaf_id = builder.create_leaf_node(modality.modality_id)

        rep_node_id = builder.create_operation_node(
            operator.__class__, [leaf_id], operator.get_current_parameters()
        )
        current_node_id = rep_node_id
        rep_dag = builder.build(current_node_id)
        dags.append(rep_dag)

        dimensionality_reduction_dags = self.add_dimensionality_reduction_operators(
            builder, current_node_id
        )
        if dimensionality_reduction_dags is not None:
            dags.extend(dimensionality_reduction_dags)

        if operator.needs_context:
            context_operators = self._get_context_operators(modality.modality_type)
            for context_op in context_operators:
                if operator.initial_context_length is not None:
                    context_length = operator.initial_context_length

                    context_node_id = builder.create_operation_node(
                        context_op,
                        [leaf_id],
                        context_op(context_length).get_current_parameters(),
                    )
                else:
                    context_node_id = builder.create_operation_node(
                        context_op,
                        [leaf_id],
                        context_op().get_current_parameters(),
                    )

                context_rep_node_id = builder.create_operation_node(
                    operator.__class__,
                    [context_node_id],
                    operator.get_current_parameters(),
                )

                agg_operator = AggregatedRepresentation(target_dimensions=1)
                context_agg_node_id = builder.create_operation_node(
                    agg_operator.__class__,
                    [context_rep_node_id],
                    agg_operator.get_current_parameters(),
                )

                dags.append(builder.build(context_agg_node_id))

        if not operator.self_contained:
            not_self_contained_reps = self._get_not_self_contained_reps(
                modality.modality_type
            )
            not_self_contained_reps = [
                rep for rep in not_self_contained_reps if rep != operator.__class__
            ]
            rep_id = current_node_id

            for rep in not_self_contained_reps:
                other_rep_id = builder.create_operation_node(
                    rep, [leaf_id], rep().get_current_parameters()
                )
                for combination in self._combination_operators:
                    combine_id = builder.create_operation_node(
                        combination.__class__,
                        [rep_id, other_rep_id],
                        combination.get_current_parameters(),
                    )
                    rep_dag = builder.build(combine_id)
                    dags.append(rep_dag)
                    if modality.modality_type in [
                        ModalityType.EMBEDDING,
                        ModalityType.IMAGE,
                        ModalityType.AUDIO,
                    ]:
                        dags.extend(
                            self.default_context_operators(
                                modality, builder, leaf_id, rep_dag, False
                            )
                        )
                    elif modality.modality_type == ModalityType.TIMESERIES:
                        dags.extend(
                            self.temporal_context_operators(
                                modality,
                                builder,
                                leaf_id,
                            )
                        )
                rep_id = combine_id

        if rep_dag.nodes[-1].operation().output_modality_type in [
            ModalityType.EMBEDDING
        ]:
            dags.extend(
                self.default_context_operators(
                    modality, builder, leaf_id, rep_dag, True
                )
            )

        return dags

    def _aggregation_needed(self, dag: RepresentationDag) -> bool:
        for modality in self.modalities:
            if modality.modality_id == dag.nodes[0].modality_id:
                last_stats = modality.stats
                break
        for node in dag.nodes[1:]:
            last_stats = node.operation(params=node.parameters).get_output_stats(
                last_stats
            )
        return len(last_stats.output_shape) > 1

    def add_aggregation_operator(self, builder, dags):
        new_dags = []
        if self._tasks_require_same_dims and self.expected_dimensions == 1:
            aggregated_dags = []
            for dag in dags:
                if self._aggregation_needed(dag):
                    agg_op = AggregatedRepresentation(
                        target_dimensions=self.expected_dimensions
                    )
                    agg_node_id = builder.create_operation_node(
                        agg_op.__class__,
                        [dag.root_node_id],
                        agg_op.get_current_parameters(),
                    )
                    aggregated_dags.append(builder.build(agg_node_id))
                else:
                    aggregated_dags.append(dag)
            new_dags = aggregated_dags
        else:
            new_dags = dags
        return new_dags

    def default_context_operators(
        self, modality, builder, leaf_id, rep_dag, apply_context_to_leaf=False
    ):
        dags = []
        if apply_context_to_leaf:
            if (
                modality.modality_type != ModalityType.TEXT
                and modality.modality_type != ModalityType.VIDEO
                and modality.modality_type != ModalityType.IMAGE
            ):
                context_operators = self._get_context_operators(modality.modality_type)
                for context_op in context_operators:
                    context_node_id = builder.create_operation_node(
                        context_op,
                        [leaf_id],
                        context_op().get_current_parameters(),
                    )
                    dags.append(builder.build(context_node_id))

        context_operators = self._get_context_operators(
            rep_dag.nodes[-1].operation().output_modality_type
        )
        for context_op in context_operators:
            context_node_id = builder.create_operation_node(
                context_op,
                [rep_dag.nodes[-1].node_id],
                context_op().get_current_parameters(),
            )
            dags.append(builder.build(context_node_id))

        return dags

    def temporal_context_operators(self, modality, builder, leaf_id):
        aggregators = self.operator_registry.get_context_representations(
            modality.modality_type
        )
        context_operators = self._get_context_operators(modality.modality_type)

        dags = []
        for agg in aggregators:
            for context_operator in context_operators:
                context_node_id = builder.create_operation_node(
                    context_operator,
                    [leaf_id],
                    context_operator(agg()).get_current_parameters(),
                )
                dags.append(builder.build(context_node_id))

        return dags


class UnimodalResults:
    def __init__(
        self,
        modalities,
        tasks,
        debug=False,
        store_cache=True,
        k=-1,
        metric_name="accuracy",
    ):
        self.modality_ids = [modality.modality_id for modality in modalities]
        self.task_names = [task.model.name for task in tasks]
        self.results = {}
        self.debug = debug
        self.cache = {}
        self.store_cache = store_cache
        self.k = k
        self.metric_name = metric_name
        for modality in self.modality_ids:
            self.results[modality] = {task_name: [] for task_name in self.task_names}
            self.cache[modality] = {task_name: [] for task_name in self.task_names}

    def add_task_result(self, task_result: ResultEntry):
        task_name = self.task_names[
            task_result.dag.nodes[-1].parameters.get("_task_idx", 0)
        ]
        self.results[task_result.dag.nodes[0].modality_id][task_name].append(
            task_result
        )
        # TODO: Take care of modality cache in executor
        if (
            self.store_cache
            and task_result.val_score[self.metric_name]
            > self.cache[task_result.dag.nodes[0].modality_id][task_name][-1].val_score[
                self.metric_name
            ]
        ):
            self.cache[task_result.dag.nodes[0].modality_id][task_name].append(
                task_result
            )
            self.cache[task_result.dag.nodes[0].modality_id][task_name] = self.cache[
                task_result.dag.nodes[0].modality_id
            ][task_name][: self.k]
        if self.debug:
            print(f"{task_result.dag.nodes[0].modality_id}_{task_name}: {task_result}")

    def add_result(
        self,
        scores,
        transform_time,
        task_name,
        task_time,
        dag,
        modality_id,
        modality=None,
    ):
        entry = ResultEntry(
            train_score=scores[0].average_scores,
            val_score=scores[1].average_scores,
            test_score=scores[2].average_scores,
            representation_time=transform_time,
            task_time=task_time,
            dag=dag,
        )

        scores = [
            -item.val_score[self.metric_name]
            for item in self.results[modality_id][task_name]
        ]
        pos = (
            bisect_left(scores, -entry.val_score[self.metric_name])
            if len(scores) > 0
            else 0
        )
        self.results[modality_id][task_name].insert(pos, entry)

        if self.store_cache and pos < self.k and modality is not None:
            self.cache[modality.modality_id][task_name].insert(pos, modality)
            self.cache[modality.modality_id][task_name] = self.cache[
                modality.modality_id
            ][task_name][: self.k]

        if self.debug:
            print(f"{modality.modality_id}_{task_name}: {entry}")

    def print_results(self):
        for modality in self.modality_ids:
            for task_name in self.task_names:
                for entry in self.results[modality][task_name]:
                    print(f"{modality}_{task_name}: {entry}")

    def get_k_best_results(
        self, modality, task, performance_metric_name, prune_cache=False
    ):
        """
        Get the k best results for the given modality
        :param modality: modality to get the best results for
        :param k: number of best results
        :param task: task to get the best results for
        :param performance_metric_name: name of the performance metric to use for ranking
        """

        task_results = self.results[modality.modality_id][task.model.name]

        results, sorted_indices = rank_by_tradeoff(
            task_results, performance_metric_name=performance_metric_name
        )

        results = results[: self.k]
        sorted_indices = sorted_indices[: self.k]
        task_cache = self.cache.get(modality.modality_id, {}).get(task.model.name, None)
        if not task_cache:
            cache = []
            for result in results:
                if result.dag.nodes[-1].parameters.get("_node_kind", False) == "task":
                    dag = copy.deepcopy(result.dag)
                    dag.nodes = dag.nodes[:-1]
                    dag.root_node_id = dag.nodes[-1].node_id
                    cache.append(dag.execute([modality]))

        elif isinstance(task_cache, list):
            cache = task_cache
        else:
            cache_items = list(task_cache.items()) if task_cache else []
            cache = [cache_items[i][1] for i in sorted_indices if i < len(cache_items)]

        if prune_cache:
            # Note: in case the unimodal results are loaded from a file, we need to initialize the cache for the modality and task
            if modality.modality_id not in self.operator_performance.cache:
                self.operator_performance.cache[modality.modality_id] = {}
            if (
                task.model.name
                not in self.operator_performance.cache[modality.modality_id]
            ):
                self.operator_performance.cache[modality.modality_id][
                    task.model.name
                ] = {}
            self.operator_performance.cache[modality.modality_id][
                task.model.name
            ] = cache

        return results, cache
