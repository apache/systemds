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
from typing import List, Any, Optional, Dict, Set, Tuple
from functools import lru_cache

import torch
from systemds.scuro import ModalityType
from systemds.scuro.drsearch.node_scheduler import (
    NodeResourceScheduler,
    ExternalRefCountCache,
)
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
    build_node_graph,
    compute_parent_refcounts,
    dags_to_graphviz,
    deduplicate_dags,
)
from bisect import bisect_left
from systemds.scuro.drsearch.representation_dag_visualizer import visualize_dag
from systemds.scuro.representations.context import Context
from systemds.scuro.representations.dimensionality_reduction import (
    DimensionalityReduction,
)
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.utils.memory_utility import estimate_modality_bytes


def _execute_node_worker(
    node: RepresentationNode,
    input_mods: List[Any],
    task: Any,
    rep_cache: Optional[Dict[str, Any]],
    gpu_id: Optional[int],
):
    # print(f"Executing node {node.node_id} inputs: {input_mods[0].modality_id}")
    node_operation = node.operation(params=node.parameters)
    if gpu_id is not None and hasattr(node_operation, "gpu_id"):
        node_operation.gpu_id = gpu_id

    if len(input_mods) == 1:
        if isinstance(node_operation, Context):
            return input_mods[0].context(node_operation)
        if isinstance(node_operation, DimensionalityReduction):
            return input_mods[0].dimensionality_reduction(node_operation)
        if isinstance(node_operation, AggregatedRepresentation):
            return node_operation.transform(input_mods[0])
        if isinstance(node_operation, UnimodalRepresentation):
            if rep_cache is not None and node_operation.name in rep_cache:
                return rep_cache[node_operation.name]
            return input_mods[0].apply_representation(node_operation)
        return input_mods[0].apply_representation(node_operation)

    fusion_op = node_operation
    if hasattr(fusion_op, "needs_training") and fusion_op.needs_training:
        return input_mods[0].combine_with_training(input_mods[1:], fusion_op, task)
    return input_mods[0].combine(input_mods[1:], fusion_op)


def _execute_task_worker(task: Any, data: Any, gpu_id: Optional[int]) -> Dict[str, Any]:
    # print(f"Executing task {task.model.name} on GPU {gpu_id}")
    if gpu_id is not None and hasattr(task, "model") and hasattr(task.model, "device"):
        task.model.device = torch.device(f"cuda:{gpu_id}")
    start = time.perf_counter()
    scores = task.run(data)
    end = time.perf_counter()
    return {"scores": scores, "task_time": end - start}


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

    def _create_scheduler(self):
        manager = mp.Manager()
        shared_state = manager.dict()
        lock = manager.Lock()

        return NodeResourceScheduler(shared_state=shared_state, lock=lock)

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

        manager = mp.Manager()
        scheduler = NodeResourceScheduler(
            shared_state=manager.dict(), lock=manager.Lock()
        )

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
                    scheduler=scheduler,
                ): modality
                for modality in self.modalities
            }

            for future in as_completed(future_to_modality):
                modality = future_to_modality[future]
                try:
                    results = future.result()
                    self._merge_results(results)
                    new_count = self._count_results(results.results)
                    self._checkpoint_manager.increment(modality.modality_id, new_count)
                    self._checkpoint_manager.checkpoint_if_due(
                        self.operator_performance.results, "eval_count_by_modality"
                    )
                except Exception as e:
                    print(f"Error processing modality {modality.modality_id}: {e}")
                    import traceback

                    traceback.print_exc()
                    self._checkpoint_manager.save_checkpoint(
                        self.operator_performance.results, "eval_count_by_modality", {}
                    )
                    continue

    def optimize(self):
        """Optimize representations for each modality"""
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
                    self.operator_performance.results, "eval_count_by_modality"
                )
                if self.save_all_results:
                    self.store_results(f"{modality.modality_id}_unimodal_results.pkl")
            except Exception as e:
                print(f"Error processing modality {modality.modality_id}: {e}")
                import traceback

                traceback.print_exc()
                self._checkpoint_manager.save_checkpoint(
                    self.operator_performance.results, "eval_count_by_modality", {}
                )
                raise

    def _estimate_node_resources(
        self, modality, node_map, parents, topo_order
    ) -> Tuple[Dict[str, tuple[float, float]], Dict[str, float]]:
        def _bytes_from_stats(stats, default_bytes_per_value: int = 4) -> float:
            if stats is None:
                return 0.0
            output_shape = getattr(stats, "output_shape", None)
            num_instances = max(1, int(getattr(stats, "num_instances", 1)))
            if not output_shape:
                return 0.0
            try:
                element_count = 1
                for dim in output_shape:
                    element_count *= max(1, int(dim))
                return float(num_instances * element_count * default_bytes_per_value)
            except Exception:
                return 0.0

        def _estimate_output_bytes(op, input_stats, fallback: float) -> float:
            try:
                out = float(op.estimate_output_memory_bytes(input_stats))
                if out > 0:
                    return out
            except Exception:
                pass
            try:
                out_stats = op.get_output_shape(input_stats)
            except Exception:
                return fallback
            return max(_bytes_from_stats(out_stats), fallback)

        node_resources: Dict[str, tuple[float, float]] = {}
        output_stats = {}
        output_bytes: Dict[str, float] = {}
        base_stats = modality.get_stats()
        fallback_cpu = max(float(modality.estimate_memory_bytes()), 1.0)

        for node_id in topo_order:
            node = node_map[node_id]
            if not node.inputs:
                output_stats[node_id] = base_stats
                output_bytes[node_id] = max(_bytes_from_stats(base_stats), fallback_cpu)
                node_resources[node_id] = (0.0, 0.0)
                continue

            if self._is_task_node(node):
                parent_id = node.inputs[0] if node.inputs else None
                parent_stats = output_stats.get(parent_id, base_stats)
                parent_bytes = max(
                    1.0, float(output_bytes.get(parent_id, fallback_cpu))
                )

                task_idx = int(node.parameters.get("_task_idx", 0))
                task_obj = self.tasks[task_idx]
                model_obj = getattr(task_obj, "model", None)

                output_shape = list(getattr(parent_stats, "output_shape", []) or [])
                input_dim = int(output_shape[-1]) if output_shape else 1
                n_train = len(getattr(task_obj, "train_indices", []) or [])
                n_val = 0
                if hasattr(task_obj, "cv_val_indices"):
                    for fold in getattr(task_obj, "cv_val_indices", []) or []:
                        n_val += len(fold)
                n_test = len(getattr(task_obj, "test_indices", []) or [])
                n_labels = 27
                labels = getattr(task_obj, "labels", None)
                if labels is not None and len(labels) > 0:
                    try:
                        first_label = labels[0]
                        if hasattr(first_label, "__len__") and not isinstance(
                            first_label, (str, bytes)
                        ):
                            n_labels = max(1, int(len(first_label)))
                        else:
                            n_labels = 1
                    except Exception:
                        n_labels = 27

                cpu_peak = parent_bytes
                gpu_peak = 0.0

                if model_obj is not None and hasattr(
                    model_obj, "estimate_peak_memory_bytes"
                ):
                    try:
                        mem = model_obj.estimate_peak_memory_bytes(
                            input_dim=input_dim,
                            n_train_samples=max(1, n_train),
                            n_labels=max(1, n_labels),
                            n_val_samples=max(0, n_val),
                            n_test_samples=max(0, n_test),
                        )
                        cpu_peak = float(mem.get("cpu_peak_bytes", cpu_peak))
                        gpu_peak = float(mem.get("gpu_peak_bytes", gpu_peak))

                        if hasattr(task_obj, "model") and hasattr(
                            task_obj.model, "device"
                        ):
                            dev = str(getattr(task_obj.model, "device", "cpu"))
                            if dev.startswith("cuda"):
                                gpu_peak = max(gpu_peak, 1.2 * 1024**3)

                        gpu_peak *= 1.35
                    except Exception:
                        pass

                if cpu_peak <= 0:
                    cpu_peak = parent_bytes * 2.0
                if gpu_peak < 0:
                    gpu_peak = 0.0

                output_stats[node_id] = parent_stats
                output_bytes[node_id] = parent_bytes
                node_resources[node_id] = (max(1.0, cpu_peak), max(0.0, gpu_peak))
                continue

            parent_ids = list(parents.get(node_id, set()))
            parent_stats = [
                output_stats[parent_id]
                for parent_id in parent_ids
                if parent_id in output_stats
            ]
            input_stats = parent_stats[0] if parent_stats else base_stats
            op = node.operation(params=node.parameters)
            summed_parent_bytes = (
                float(sum(output_bytes.get(parent_id, 0.0) for parent_id in parent_ids))
                if parent_ids
                else fallback_cpu
            )
            if summed_parent_bytes <= 0:
                summed_parent_bytes = fallback_cpu

            cpu_peak = 0.0
            gpu_peak = 0.0
            try:
                peak = op.estimate_peak_memory_bytes(input_stats)
                cpu_peak = float(peak.get("cpu_peak_bytes", 0.0))
                gpu_peak = float(peak.get("gpu_peak_bytes", 0.0))
            except Exception:
                pass

            try:
                output_stats[node_id] = op.get_output_shape(input_stats)
            except Exception:
                output_stats[node_id] = input_stats

            estimated_output_bytes = _estimate_output_bytes(
                op,
                input_stats,
                max(_bytes_from_stats(output_stats[node_id]), fallback_cpu * 0.1),
            )
            output_bytes[node_id] = max(1.0, float(estimated_output_bytes))

            if cpu_peak <= 0:
                cpu_peak = summed_parent_bytes + output_bytes[node_id]
            else:
                cpu_peak = max(cpu_peak, output_bytes[node_id])

            node_resources[node_id] = (cpu_peak, gpu_peak)
        return node_resources, output_bytes

    @staticmethod
    def _is_task_node(node: RepresentationNode) -> bool:
        return bool(getattr(node, "parameters", {}).get("_node_kind") == "task")

    @staticmethod
    def _is_memory_error(exception: Exception) -> bool:
        msg = str(exception).lower()
        if (
            "cuda error" in msg
            or "out of memory" in msg
            or "not enough memory" in msg
            or "cublas_status_alloc_failed" in msg
            or "cublascreate" in msg
            or "cuda error: cublas_status_alloc_failed" in msg
        ):
            return True
        return isinstance(exception, torch.cuda.OutOfMemoryError)

    def _expand_dags_with_task_roots(
        self, dags: List[RepresentationDag], completed_task_nodes: Set[str]
    ) -> Tuple[
        List[RepresentationDag],
        Dict[str, RepresentationDag],
        Dict[str, str],
        Dict[str, Set[str]],
    ]:
        expanded_dags: List[RepresentationDag] = []
        task_node_to_dag: Dict[str, RepresentationDag] = {}
        task_node_to_root: Dict[str, str] = {}
        root_to_task_nodes: Dict[str, Set[str]] = {}

        for dag in dags:
            root_id = dag.root_node_id
            for task_idx, _ in enumerate(self.tasks):
                task_node_id = f"task_{root_id}_{task_idx}"
                if task_node_id in completed_task_nodes:
                    continue

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
                task_node_to_dag[task_node_id] = dag
                task_node_to_root[task_node_id] = root_id
                root_to_task_nodes.setdefault(root_id, set()).add(task_node_id)

        return expanded_dags, task_node_to_dag, task_node_to_root, root_to_task_nodes

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
        operators = []
        for operator in modality_specific_operators:
            dags.extend(self._build_modality_dag(modality, operator()))
            operators.append(operator())

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
        dags = deduplicate_dags(dags)
        dags_to_graphviz(dags).render(
            filename=f"dags_{modality.modality_id}.png", format="png"
        )

        if skip_remaining > 0:
            dags = dags[skip_remaining:]

        rep_cache = None
        if hasattr(modality, "data_loader") and modality.data_loader.chunk_size:
            rep_cache = modality.apply_representations(operators)

        node_checkpoint = CheckpointManager(
            self.result_path or ".",
            f"unimodal_nodes_checkpoint_{modality.modality_id}_",
            checkpoint_every=self.checkpoint_every,
            resume=self.resume,
        )
        checkpoint_entries = []
        completed_roots: Set[str] = set()
        completed_task_nodes: Set[str] = set()
        loaded = node_checkpoint.load_latest()
        if loaded:
            loaded_entries, meta, _ = loaded
            checkpoint_entries = loaded_entries or []
            completed_roots = set(meta.get("completed_roots", []))
            completed_task_nodes = set(meta.get("completed_task_nodes", []))
            for result_entry in checkpoint_entries:
                local_results.add_result(
                    result_entry["scores"],
                    result_entry["transform_time"],
                    result_entry["task_name"],
                    result_entry["task_time"],
                    result_entry["dag"],
                    modality.modality_id,
                )
            node_checkpoint.eval_count = meta.get("eval_count", len(checkpoint_entries))
            node_checkpoint._last_checkpoint_eval_count = node_checkpoint.eval_count
            node_checkpoint.counts_by_key = meta.get(
                "eval_count_by_modality",
                {modality.modality_id: len(checkpoint_entries)},
            )

        dags = [dag for dag in dags if dag.root_node_id not in completed_roots]
        if completed_task_nodes:
            pending_dags = []
            for dag in dags:
                pending = any(
                    f"task_{dag.root_node_id}_{task_idx}" not in completed_task_nodes
                    for task_idx, _ in enumerate(self.tasks)
                )
                if pending:
                    pending_dags.append(dag)
                else:
                    completed_roots.add(dag.root_node_id)
            dags = pending_dags
        if not dags:
            return local_results

        expanded_dags, task_node_to_dag, task_node_to_root, root_to_task_nodes = (
            self._expand_dags_with_task_roots(dags, completed_task_nodes)
        )

        if not expanded_dags:
            return local_results
        node_map, children, parents, roots, leaves, topo_order = build_node_graph(
            expanded_dags
        )
        task_node_ids = set(task_node_to_root.keys())
        if not task_node_ids:
            return local_results
        parent_remaining = compute_parent_refcounts(children)
        unresolved_parents = {}
        for node_id in node_map:
            unresolved_parents[node_id] = sum(
                1
                for parent_id in parents.get(node_id, set())
                if parent_id not in leaves
            )

        node_resources, estimated_output_bytes = self._estimate_node_resources(
            modality, node_map, parents, topo_order
        )
        max_workers = max(1, min(len(node_map), mp.cpu_count()))

        if scheduler is None:
            scheduler = NodeResourceScheduler()

        ready_nodes = [
            node_id
            for node_id in topo_order
            if node_id not in leaves and unresolved_parents.get(node_id, 0) == 0
        ]
        running_nodes = {}
        node_results = {leaf_id: modality for leaf_id in leaves}
        external_cache = ExternalRefCountCache(scheduler)
        memory_retry_counts: Dict[str, int] = {}
        max_memory_retries = 25

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            while ready_nodes or running_nodes:
                scheduled_any = False
                for node_id in list(ready_nodes):
                    cpu_mem, gpu_mem = node_resources.get(node_id, (1.0, 0.0))
                    ok, gpu_id = scheduler.reserve_exec(cpu_mem, gpu_mem)
                    if not ok:
                        continue
                    node = node_map[node_id]
                    input_mods = []
                    all_available = True
                    for parent_id in node.inputs:
                        parent_value = external_cache.get(parent_id)
                        if parent_value is None:
                            parent_value = node_results.get(parent_id)
                        if parent_value is None:
                            all_available = False
                            break
                        input_mods.append(parent_value)

                    if not all_available:
                        scheduler.release_exec(cpu_mem, gpu_mem, gpu_id)
                        continue

                    if not scheduler.can_execute_now(cpu_mem, gpu_mem, gpu_id):
                        scheduler.release_exec(cpu_mem, gpu_mem, gpu_id)
                        continue

                    if self._is_task_node(node):
                        task_idx = int(node.parameters.get("_task_idx", 0))
                        future = executor.submit(
                            _execute_task_worker,
                            self.tasks[task_idx],
                            input_mods[0].data if input_mods else None,
                            gpu_id,
                        )
                    else:
                        future = executor.submit(
                            _execute_node_worker,
                            node,
                            input_mods,
                            None,
                            rep_cache,
                            gpu_id,
                        )
                    running_nodes[future] = (node_id, cpu_mem, gpu_mem, gpu_id)
                    ready_nodes.remove(node_id)
                    scheduled_any = True

                if not running_nodes:
                    break

                done = next(as_completed(running_nodes), None)
                if done is None:
                    break
                node_id, cpu_mem, gpu_mem, gpu_id = running_nodes.pop(done)
                scheduler.release_exec(cpu_mem, gpu_mem, gpu_id)
                try:
                    node = node_map[node_id]
                    node_children = children.get(node_id, set())
                    if self._is_task_node(node):
                        task_payload = done.result()
                        task_idx = int(node.parameters.get("_task_idx", 0))
                        task = self.tasks[task_idx]
                        root_id = task_node_to_root.get(node_id)
                        dag = task_node_to_dag.get(node_id)
                        parent_modality = None
                        if node.inputs:
                            parent_id = node.inputs[0]
                            parent_modality = external_cache.get(parent_id)
                            if parent_modality is None:
                                parent_modality = node_results.get(parent_id)
                        result_entry = {
                            "scores": task_payload["scores"],
                            "transform_time": getattr(
                                parent_modality, "transform_time", 0.0
                            ),
                            "task_name": task.model.name,
                            "task_time": task_payload["task_time"],
                            "dag": dag,
                            "modality_id": modality.modality_id,
                        }
                        checkpoint_entries.append(result_entry)
                        node_checkpoint.increment(modality.modality_id, 1)
                        completed_task_nodes.add(node_id)
                        if root_id and root_id in root_to_task_nodes:
                            if root_to_task_nodes[root_id].issubset(
                                completed_task_nodes
                            ):
                                completed_roots.add(root_id)
                        node_checkpoint.checkpoint_if_due(
                            checkpoint_entries,
                            "eval_count_by_modality",
                            {
                                "completed_roots": list(completed_roots),
                                "completed_task_nodes": list(completed_task_nodes),
                            },
                        )
                        local_results.add_result(
                            result_entry["scores"],
                            result_entry["transform_time"],
                            result_entry["task_name"],
                            result_entry["task_time"],
                            result_entry["dag"],
                            modality.modality_id,
                        )
                        if (
                            self.debug
                            and root_id in completed_roots
                            and dag is not None
                        ):
                            visualize_dag(dag)
                    else:
                        result_modality = done.result()
                        node_results[node_id] = result_modality

                        if len(node_children) > 0:
                            result_mem = max(
                                1.0, float(estimate_modality_bytes(result_modality))
                            )
                            if result_mem <= 1.0:
                                result_mem = max(
                                    1.0,
                                    float(
                                        estimated_output_bytes.get(node_id, result_mem)
                                    ),
                                )
                            external_cache.put(
                                node_id,
                                result_modality,
                                len(node_children),
                                result_mem,
                                0.0,
                                gpu_id,
                            )
                            if external_cache.get(node_id) is not None:
                                node_results.pop(node_id, None)

                    for parent_id in node_map[node_id].inputs:
                        parent_remaining[parent_id] = (
                            parent_remaining.get(parent_id, 0) - 1
                        )
                        if parent_remaining[parent_id] <= 0:
                            external_cache.evict(parent_id)
                            node_results.pop(parent_id, None)

                    for child_id in node_children:
                        unresolved_parents[child_id] -= 1
                        if unresolved_parents[child_id] == 0:
                            ready_nodes.append(child_id)
                except Exception as e:
                    if self._is_memory_error(e):
                        retries = memory_retry_counts.get(node_id, 0) + 1
                        memory_retry_counts[node_id] = retries
                        if retries <= max_memory_retries:
                            if node_id not in ready_nodes:
                                ready_nodes.append(node_id)
                            time.sleep(0.05)
                            continue
                        print(
                            f"Node {node_id} exceeded memory retries ({max_memory_retries}) "
                            f"for modality {modality.modality_id}"
                        )
                    print(
                        f"Error processing node {node_id} for modality {modality.modality_id}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                if not scheduled_any and not running_nodes and ready_nodes:
                    time.sleep(0.01)

        if completed_roots or completed_task_nodes:
            node_checkpoint.save_checkpoint(
                checkpoint_entries,
                "eval_count_by_modality",
                {
                    "completed_roots": list(completed_roots),
                    "completed_task_nodes": list(completed_task_nodes),
                },
            )
            node_checkpoint.cleanup()

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

        # for modality in self.modalities:
        #     for task_name in local_results.cache[modality]:
        #         for key, value in local_results.cache[modality][task_name].items():
        #             self.operator_performance.cache[modality][task_name][key] = value

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

                agg_operator = AggregatedRepresentation()
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
            last_stats = node.operation().get_output_shape(last_stats)
        return len(last_stats.output_shape) > 1 or last_stats.num_instances > 1

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
            cache = [results[i].dag.execute([modality]) for i in range(len(results))]
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


@dataclass
class ResultEntry:
    val_score: PerformanceMeasure = None
    train_score: PerformanceMeasure = None
    test_score: PerformanceMeasure = None
    representation_time: float = 0.0
    task_time: float = 0.0
    dag: RepresentationDag = None
    tradeoff_score: float = 0.0
