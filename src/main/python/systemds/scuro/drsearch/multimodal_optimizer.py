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
import os
import multiprocessing as mp
import itertools
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Generator
from systemds.scuro.drsearch.task import Task, PerformanceMeasure
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationDAGBuilder,
)
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.utils.schema_helpers import get_shape

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import pickle
import copy
import time
import traceback
from itertools import chain


def _evaluate_dag_worker(dag_pickle, task_pickle, modalities_pickle, debug=False):
    try:
        dag = pickle.loads(dag_pickle)
        task = pickle.loads(task_pickle)
        modalities_for_dag = pickle.loads(modalities_pickle)

        start_time = time.time()
        if debug:
            print(
                f"[DEBUG][worker] pid={os.getpid()} evaluating dag_root={getattr(dag, 'root_node_id', None)} task={getattr(task.model, 'name', None)}"
            )

        dag_copy = copy.deepcopy(dag)
        task_copy = copy.deepcopy(task)

        fused_representation = dag_copy.execute(modalities_for_dag, task_copy)
        if fused_representation is None:
            return None

        final_representation = fused_representation[
            list(fused_representation.keys())[-1]
        ]
        from systemds.scuro.utils.schema_helpers import get_shape
        from systemds.scuro.representations.aggregated_representation import (
            AggregatedRepresentation,
        )
        from systemds.scuro.representations.aggregate import Aggregation

        if task_copy.expected_dim == 1 and get_shape(final_representation.metadata) > 1:
            agg_operator = AggregatedRepresentation(Aggregation())
            final_representation = agg_operator.transform(final_representation)

        eval_start = time.time()
        scores = task_copy.run(final_representation.data)
        eval_time = time.time() - eval_start
        total_time = time.time() - start_time

        return OptimizationResult(
            dag=dag_copy,
            train_score=scores[0].average_scores,
            val_score=scores[1].average_scores,
            runtime=total_time,
            task_name=task_copy.model.name,
            task_time=eval_time,
            representation_time=total_time - eval_time,
        )
    except Exception:
        if debug:
            traceback.print_exc()
        return None


class MultimodalOptimizer:
    def __init__(
        self,
        modalities: List[Any],
        unimodal_optimization_results: Any,
        tasks: List[Any],
        k: int = 2,
        debug: bool = True,
        min_modalities: int = 2,
        max_modalities: int = None,
        metric: str = "accuracy",
    ):
        self.modalities = modalities
        self.tasks = tasks
        self.k = k
        self.debug = debug
        self.min_modalities = max(2, min_modalities)
        self.max_modalities = max_modalities or len(modalities)

        self.operator_registry = Registry()
        self.fusion_operators = self.operator_registry.get_fusion_operators()
        self.metric_name = metric

        self.k_best_representations = self._extract_k_best_representations(
            unimodal_optimization_results
        )
        self.optimization_results = []

    def optimize_parallel(
        self, max_combinations: int = None, max_workers: int = 2, batch_size: int = 4
    ) -> Dict[str, List["OptimizationResult"]]:
        all_results = {}

        for task in self.tasks:
            task_copy = copy.deepcopy(task)
            if self.debug:
                print(
                    f"[DEBUG] Optimizing multimodal fusion for task: {task.model.name}"
                )
            all_results[task.model.name] = []
            evaluated_count = 0
            outstanding = set()
            stop_generation = False

            modalities_for_task = list(
                chain.from_iterable(
                    self.k_best_representations[task.model.name].values()
                )
            )
            task_pickle = pickle.dumps(task_copy)
            modalities_pickle = pickle.dumps(modalities_for_task)
            ctx = mp.get_context("spawn")
            start = time.time()
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                for modality_subset in self._generate_modality_combinations():
                    if stop_generation:
                        break
                    if self.debug:
                        print(f"[DEBUG] Evaluating modality subset: {modality_subset}")

                    for repr_combo in self._generate_representation_combinations(
                        modality_subset, task.model.name
                    ):
                        if stop_generation:
                            break

                        for dag in self._generate_fusion_dags(
                            modality_subset, repr_combo
                        ):
                            if max_combinations and evaluated_count >= max_combinations:
                                stop_generation = True
                                break

                            dag_pickle = pickle.dumps(dag)
                            fut = ex.submit(
                                _evaluate_dag_worker,
                                dag_pickle,
                                task_pickle,
                                modalities_pickle,
                                self.debug,
                            )
                            outstanding.add(fut)

                            if len(outstanding) >= batch_size:
                                done, not_done = wait(
                                    outstanding, return_when=FIRST_COMPLETED
                                )
                                for fut_done in done:
                                    try:
                                        result = fut_done.result()
                                        if result is not None:
                                            all_results[task.model.name].append(result)
                                    except Exception:
                                        if self.debug:
                                            traceback.print_exc()
                                    evaluated_count += 1
                                    if self.debug and evaluated_count % 100 == 0:
                                        print(
                                            f"[DEBUG] Evaluated {evaluated_count} combinations..."
                                        )
                                    else:
                                        print(".", end="")
                                outstanding = set(not_done)

                    break

                if outstanding:
                    done, not_done = wait(outstanding)
                    for fut_done in done:
                        try:
                            result = fut_done.result()
                            if result is not None:
                                all_results[task.model.name].append(result)
                        except Exception:
                            if self.debug:
                                traceback.print_exc()
                        evaluated_count += 1
                        if self.debug and evaluated_count % 100 == 0:
                            print(
                                f"[DEBUG] Evaluated {evaluated_count} combinations..."
                            )
                        else:
                            print(".", end="")
            end = time.time()
            if self.debug:
                print(f"\n[DEBUG] Total optimization time: {end-start}")
                print(
                    f"[DEBUG] Task completed: {len(all_results[task.model.name])} valid combinations evaluated"
                )

        self.optimization_results = all_results

        if self.debug:
            print(f"[DEBUG] Optimization completed")

        return all_results

    def _extract_k_best_representations(
        self, unimodal_optimization_results: Any
    ) -> Dict[str, Dict[str, List[Any]]]:
        k_best = {}

        for task in self.tasks:
            k_best[task.model.name] = {}

            for modality in self.modalities:
                k_best_results, cached_data = (
                    unimodal_optimization_results.get_k_best_results(
                        modality, self.k, task, self.metric_name
                    )
                )

                k_best[task.model.name][modality.modality_id] = cached_data

        return k_best

    def _generate_modality_combinations(self) -> Generator[List[str], None, None]:
        modality_ids = [mod.modality_id for mod in self.modalities]

        for r in range(
            self.min_modalities, min(self.max_modalities + 1, len(modality_ids) + 1)
        ):
            for modality_subset in itertools.combinations(modality_ids, r):
                yield list(modality_subset)

    def _generate_representation_combinations(
        self, modality_subset: List[str], task_name: str
    ) -> Generator[Dict[str, int], None, None]:
        representation_options = []

        for modality_id in modality_subset:
            num_representations = len(
                self.k_best_representations[task_name][modality_id]
            )
            representation_options.append(list(range(num_representations)))

        for combo in itertools.product(*representation_options):
            yield {
                modality_id: repr_idx
                for modality_id, repr_idx in zip(modality_subset, combo)
            }

    def _generate_fusion_dags(
        self, modality_subset: List[str], representation_combo: Dict[str, int]
    ) -> Generator[RepresentationDag, None, None]:
        leaf_infos = [(m, representation_combo[m]) for m in modality_subset]

        def gen_trees(indices: List[int]):
            if len(indices) == 1:
                yield indices[0]
                return
            for split in range(1, len(indices)):
                for left_idxs in itertools.combinations(indices, split):
                    left = list(left_idxs)
                    right = [i for i in indices if i not in left]
                    for l_tree in gen_trees(left):
                        for r_tree in gen_trees(right):
                            yield (l_tree, r_tree)

        def build_variants(
            subtree, base_builder: RepresentationDAGBuilder, leaf_id_map
        ):
            variants = []

            if isinstance(subtree, int):
                variants.append((base_builder, leaf_id_map[subtree]))
                return variants

            left_sub, right_sub = subtree

            left_variants = build_variants(
                left_sub, copy.deepcopy(base_builder), leaf_id_map
            )

            for left_builder, left_root in left_variants:
                right_variants = build_variants(
                    right_sub, copy.deepcopy(left_builder), leaf_id_map
                )

                for right_builder, right_root in right_variants:
                    for fusion_op_class in self.fusion_operators:
                        new_builder = copy.deepcopy(right_builder)
                        fusion_op = fusion_op_class()
                        fusion_id = new_builder.create_operation_node(
                            fusion_op.__class__,
                            [left_root, right_root],
                            fusion_op.get_current_parameters(),
                        )
                        variants.append((new_builder, fusion_id))

            return variants

        n = len(leaf_infos)

        for permuted_leaf_infos in itertools.permutations(leaf_infos, n):
            base_builder = RepresentationDAGBuilder()
            leaf_id_map = {}
            for idx, (modality_id, repr_idx) in enumerate(permuted_leaf_infos):
                nodeid = base_builder.create_leaf_node(modality_id, repr_idx)
                leaf_id_map[idx] = nodeid

            indices = list(range(n))

            for tree in gen_trees(indices):
                variants = build_variants(tree, base_builder, leaf_id_map)
                for builder_variant, root_id in variants:
                    try:
                        yield builder_variant.build(root_id)
                    except ValueError:
                        if self.debug:
                            print(f"[DEBUG] Skipping invalid DAG for root {root_id}")
                        continue

    def _evaluate_dag(self, dag: RepresentationDag, task: Task) -> "OptimizationResult":
        start_time = time.time()
        try:
            tid = threading.get_ident()
            tname = threading.current_thread().name

            dag_copy = copy.deepcopy(dag)
            modalities_for_dag = copy.deepcopy(
                list(
                    chain.from_iterable(
                        self.k_best_representations[task.model.name].values()
                    )
                )
            )
            task_copy = copy.deepcopy(task)
            fused_representation = dag_copy.execute(
                modalities_for_dag,
                task_copy,
            )

            if fused_representation is None:
                return None

            final_representation = fused_representation[
                list(fused_representation.keys())[-1]
            ]
            if (
                task_copy.expected_dim == 1
                and get_shape(final_representation.metadata) > 1
            ):
                agg_operator = AggregatedRepresentation(Aggregation())
                final_representation = agg_operator.transform(final_representation)

            eval_start = time.time()
            scores = task_copy.run(final_representation.data)
            eval_time = time.time() - eval_start

            total_time = time.time() - start_time

            return OptimizationResult(
                dag=dag_copy,
                train_score=scores[0].average_scores,
                val_score=scores[1].average_scores,
                runtime=total_time,
                representation_time=total_time - eval_time,
                task_name=task_copy.model.name,
                task_time=eval_time,
            )

        except Exception as e:
            print(f"Error evaluating DAG: {e}")
            traceback.print_exc()
            return None

    def _get_modality_by_id_and_instance_id(self, modalities, modality_id, instance_id):
        counter = 0
        for modality in modalities:
            if modality.modality_id == modality_id:
                if counter == instance_id or instance_id == -1:
                    return modality
                else:
                    counter += 1
        return None

    def optimize(
        self, max_combinations: int = None
    ) -> Dict[str, List["OptimizationResult"]]:
        all_results = {}

        for task in self.tasks:
            if self.debug:
                print(
                    f"[DEBUG] Optimizing multimodal fusion for task: {task.model.name}"
                )
            all_results[task.model.name] = []
            evaluated_count = 0

            for modality_subset in self._generate_modality_combinations():
                if self.debug:
                    print(f"[DEBUG] Evaluating modality subset: {modality_subset}")

                for repr_combo in self._generate_representation_combinations(
                    modality_subset, task.model.name
                ):

                    for dag in self._generate_fusion_dags(modality_subset, repr_combo):
                        if max_combinations and evaluated_count >= max_combinations:
                            break

                        result = self._evaluate_dag(dag, task)
                        if result is not None:
                            all_results[task.model.name].append(result)

                        evaluated_count += 1

                        if self.debug and evaluated_count % 100 == 0:
                            print(f"    Evaluated {evaluated_count} combinations...")

                    if max_combinations and evaluated_count >= max_combinations:
                        break

                if max_combinations and evaluated_count >= max_combinations:
                    break

            if self.debug:
                print(
                    f"[DEBUG] Task completed: {len(all_results[task.model.name])} valid combinations evaluated"
                )

        self.optimization_results = all_results

        if self.debug:
            print(f"[DEBUG] Optimization completed")

        return all_results

    def store_results(self, file_name=None):
        if file_name is None:
            import time

            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = "multimodal_optimizer" + timestr + ".pkl"

        with open(file_name, "wb") as f:
            pickle.dump(self.optimization_results, f)


@dataclass
class OptimizationResult:
    dag: RepresentationDag
    train_score: PerformanceMeasure = None
    val_score: PerformanceMeasure = None
    runtime: float = 0.0
    task_time: float = 0.0
    representation_time: float = 0.0
    task_name: str = ""
    tradeoff_score: float = 0.0
