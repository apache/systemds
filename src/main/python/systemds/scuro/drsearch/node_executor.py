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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import os
from multiprocessing import shared_memory
from systemds.scuro import Modality
from systemds.scuro.drsearch.modality_shared_memory import add_shared_memory_candidate
from systemds.scuro.drsearch.node_scheduler import MemoryAwareNodeScheduler
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationNode,
)

import numpy as np
from typing import Any, Dict, List, Optional
import multiprocessing as mp
import psutil
import time
import torch
from systemds.scuro.drsearch.task import PerformanceMeasure
from systemds.scuro.representations.context import Context
from systemds.scuro.representations.dimensionality_reduction import (
    DimensionalityReduction,
)
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.utils.checkpointing import CheckpointManager
import threading
import time
import psutil
import os
from systemds.scuro.utils.static_variables import DEBUG


def measure_peak_rss_during(fn, *args, sample_s=0.01, **kwargs):
    proc = psutil.Process(os.getpid())
    baseline = proc.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sampler():
        nonlocal peak
        while not stop.is_set():
            rss = proc.memory_info().rss
            if rss > peak:
                peak = rss
            time.sleep(sample_s)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        out = fn(*args, **kwargs)
    finally:
        stop.set()
        t.join()

    return out, (peak - baseline), peak


class RefCountResultCache:
    def __init__(self):
        self.cache = {}
        self.ref_count = {}
        self.memory_usage_per_node = {}
        self.shared_memory_names = {}

    def get(self, node_id: str) -> Any:
        return self.cache[node_id]

    def add_result(self, node_id: str, result: Any):
        resident_bytes = result.calculate_memory_usage()
        shared_backing_bytes = 0

        if hasattr(result, "data"):
            try:
                data, shm_name, data_nbytes, resident_bytes = (
                    add_shared_memory_candidate(result.data, resident_bytes)
                )
                if data is not None:
                    result.data = data
                    self.shared_memory_names[node_id] = [shm_name]
                    shared_backing_bytes = data_nbytes
            except Exception as e:
                print(
                    f"Failed to move cache entry {node_id} to shared memory, falling back to RAM: {e}"
                )

        self.cache[node_id] = result
        self.memory_usage_per_node[node_id] = int(resident_bytes)
        if DEBUG:
            print(
                f"Node {node_id} has a CPU memory usage of {self.memory_usage_per_node[node_id]/1024**3:.5f} GB"
                + (
                    f" (shared-memory backing: {shared_backing_bytes/1024**3:.5f} GB)"
                    if shared_backing_bytes > 0
                    else ""
                )
            )

    def inc_ref(self, node_id: str):
        if node_id not in self.ref_count:
            self.ref_count[node_id] = 0
        self.ref_count[node_id] += 1

    def dec_ref(self, node_id: str):
        self.ref_count[node_id] -= 1
        if self.ref_count[node_id] == 0:
            del self.cache[node_id]
            del self.ref_count[node_id]
            del self.memory_usage_per_node[node_id]
            self._cleanup_shared_memory(node_id)

    def clear(self, node_id: str):
        if node_id in self.cache:
            del self.cache[node_id]
        if node_id in self.ref_count:
            del self.ref_count[node_id]
        if node_id in self.memory_usage_per_node:
            del self.memory_usage_per_node[node_id]
        self._cleanup_shared_memory(node_id)

    def __len__(self):
        return len(self.cache)

    def get_memory_total_memory_usage(self):
        return sum(self.memory_usage_per_node.values())

    def _cleanup_shared_memory(self, node_id: str):
        names = self.shared_memory_names.pop(node_id, [])
        for shm_name in names:
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def cleanup_all(self):
        for node_id in list(self.shared_memory_names.keys()):
            self._cleanup_shared_memory(node_id)


def _execute_node_worker(node, input_mods, task, rep_cache, gpu_id):
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    node_operation = node.operation(params=node.parameters)
    operation_name = node_operation.name
    if DEBUG:
        print(f"Executing node {node.node_id} {operation_name} on GPU {gpu_id}")

    if gpu_id is not None and hasattr(node_operation, "gpu_id"):
        node_operation.gpu_id = gpu_id

    def _run_node_op():
        if len(input_mods) == 1:
            if isinstance(node_operation, Context):
                return input_mods[0].context(node_operation)
            elif isinstance(node_operation, DimensionalityReduction):
                return input_mods[0].dimensionality_reduction(node_operation)
            elif isinstance(node_operation, AggregatedRepresentation):
                return node_operation.transform(input_mods[0])
            elif isinstance(node_operation, UnimodalRepresentation):
                pushdown_config = node.parameters.get("_pushdown_aggregation", None)
                agg = None
                if pushdown_config is not None:
                    agg = AggregatedRepresentation(params=pushdown_config)
                if rep_cache is not None and node_operation.name in rep_cache:
                    return rep_cache[node_operation.name]
                return input_mods[0].apply_representation(
                    node_operation, aggregation=agg
                )
            return input_mods[0].apply_representation(node_operation)
        else:
            fusion_op = node_operation
            if hasattr(fusion_op, "needs_training") and fusion_op.needs_training:
                return input_mods[0].combine_with_training(
                    input_mods[1:], fusion_op, task
                )
            return input_mods[0].combine(input_mods[1:], fusion_op)

    result, peak_delta_bytes, peak_abs_rss = measure_peak_rss_during(
        _run_node_op,
        sample_s=0.01,
    )

    gpu_peak_bytes = (
        torch.cuda.max_memory_allocated(device) if gpu_id is not None else 0
    )

    return {
        "result": result,
        "peak_bytes": peak_delta_bytes,
        "peak_abs_rss_bytes": peak_abs_rss,
        "gpu_peak_bytes": gpu_peak_bytes,
        "operation_name": operation_name,
    }


def _execute_task_worker(
    task_node_id: str, task: Any, data: Any, gpu_id: Optional[int]
) -> Dict[str, Any]:

    if DEBUG:
        print(f"Executing task {task_node_id} on GPU {gpu_id}")
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    if gpu_id is not None and hasattr(task, "model") and hasattr(task.model, "device"):
        task.model.device = torch.device(f"cuda:{gpu_id}")

    def _run_task():
        start = time.perf_counter()
        scores = task.run(data)
        end = time.perf_counter()
        return scores, end - start

    gpu_peak_bytes = (
        torch.cuda.max_memory_allocated(device) if gpu_id is not None else 0
    )
    result, peak_delta_bytes, peak_abs_rss = measure_peak_rss_during(
        _run_task,
        sample_s=0.01,
    )
    if DEBUG:
        print(
            f"Task {task_node_id} has a CPU peak memory usage of {peak_delta_bytes/1024**3:.2f} GB, and a GPU peak memory usage of {gpu_peak_bytes/1024**3:.2f} GB"
        )
    return {
        "scores": result[0],
        "task_time": result[1],
        "peak_bytes": peak_delta_bytes,
        "gpu_peak_bytes": gpu_peak_bytes,
    }


class NodeExecutor:
    def __init__(
        self,
        dags: List[RepresentationDag],
        modalities: List[Modality],
        tasks: List[Any],
        checkpoint_manager: Optional[CheckpointManager] = None,
        max_num_workers: int = -1,
        result_path: Optional[str] = None,
    ):
        available_total_cpu = (
            float(psutil.virtual_memory().available)
            - float(psutil.virtual_memory().available) * 0.30
        )
        self.dags = dags
        self.scheduler = MemoryAwareNodeScheduler(
            dags, modalities, tasks, available_total_cpu
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=result_path if result_path is not None else os.getcwd(),
            prefix=f"node_executor_checkpoint_{modalities[0].modality_id}_",
            checkpoint_every=1,
            resume=False,
        )
        self.max_num_workers = (
            min(mp.cpu_count(), max_num_workers)
            if max_num_workers != -1
            else mp.cpu_count()
        )
        self.modalities = modalities
        self.tasks = tasks
        self.result_cache = RefCountResultCache()
        self.memory_usage_checkpoint = CheckpointManager(
            checkpoint_dir=result_path if result_path is not None else os.getcwd(),
            prefix=f"memory_usage_checkpoint_{modalities[0].modality_id}_",
            checkpoint_every=1,
            resume=False,
        )

    def run(self) -> None:
        task_results = {}
        memory_usage_data = {}

        self._materialize_leaf_modalities_in_shared_memory()

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=self.max_num_workers, mp_context=ctx
        ) as executor:
            future_to_node_id = {}

            def submit_node(node_id: str):
                node = self.scheduler.mapping[node_id]
                gpu_id = node.gpu_id
                parent_result = None
                parent_node_id = self.scheduler.get_valid_parent(node_id)
                if parent_node_id is not None:
                    parent_result = self.result_cache.get(parent_node_id)
                if self._is_task_node(node):
                    task_result = ResultEntry(
                        dag=self._get_dag_from_node_ids(node_id),
                        representation_time=parent_result.transform_time,
                    )
                    task_results[node_id] = task_result
                    task_idx = int(node.parameters.get("_task_idx", 0))
                    future = executor.submit(
                        _execute_task_worker,
                        node_id,
                        self.tasks[task_idx],
                        (
                            self.modalities[0].data
                            if parent_result is None
                            else parent_result.data
                        ),
                        gpu_id,
                    )
                else:
                    future = executor.submit(
                        _execute_node_worker,
                        node,
                        self.modalities if parent_result is None else [parent_result],
                        None,
                        None,
                        gpu_id,
                    )
                self.scheduler.move_to_running(node_id)
                future_to_node_id[future] = node_id

            def submit_new_ready_nodes():
                for node_id in self.scheduler.get_runnable().copy():
                    submit_node(node_id)

            submit_new_ready_nodes()

            while future_to_node_id or not self.scheduler.is_finished():
                if not future_to_node_id:
                    submit_new_ready_nodes()
                    continue

                done, _ = wait(
                    set(future_to_node_id.keys()), return_when=FIRST_COMPLETED
                )

                for future in done:
                    node_id = future_to_node_id.pop(future)
                    try:
                        result = future.result()
                    except Exception as e:
                        err_cls = type(e)
                        err_mod = err_cls.__module__
                        if err_mod.startswith("torch"):
                            torch.cuda.empty_cache()
                        print(f"Error executing node {node_id}: {e}")
                        self.scheduler.add_failed_node(node_id)
                        continue

                    peak_bytes = result["peak_bytes"]
                    gpu_peak_bytes = result["gpu_peak_bytes"]

                    node = self.scheduler.mapping[node_id]
                    if self._is_task_node(node):
                        task_results[node_id].task_time = result["task_time"]
                        task_results[node_id].train_score = result["scores"][
                            0
                        ].average_scores
                        task_results[node_id].val_score = result["scores"][
                            1
                        ].average_scores
                        task_results[node_id].test_score = result["scores"][
                            2
                        ].average_scores
                        self.checkpoint_manager.increment(node_id)
                        self.checkpoint_manager.checkpoint_if_due(task_results)
                        self._checkpoint_memory_usage(
                            node_id,
                            peak_bytes,
                            gpu_peak_bytes,
                            "task",
                            memory_usage_data,
                            None,
                        )

                        parent_node_id = self.scheduler.get_valid_parent(node_id)
                        if parent_node_id is not None:
                            self.result_cache.dec_ref(parent_node_id)
                            if (
                                parent_node_id in self.result_cache.ref_count
                                and self.result_cache.ref_count[parent_node_id] == 0
                            ):
                                self.result_cache.clear(parent_node_id)
                        self.scheduler.complete_node(node_id)

                    else:
                        transformed_modality = result["result"]
                        actual_stats = self._infer_actual_output_stats(
                            transformed_modality
                        )
                        estimated_stats = self.scheduler.node_stats.get(node_id)

                        if actual_stats is not None and (
                            estimated_stats is None
                            or not getattr(
                                estimated_stats, "output_shape_is_known", True
                            )
                        ):
                            self.scheduler.update_node_stats_and_reestimate_descendants(
                                node_id, actual_stats
                            )
                        self._checkpoint_memory_usage(
                            node_id,
                            peak_bytes,
                            gpu_peak_bytes,
                            result["operation_name"],
                            memory_usage_data,
                            transformed_modality.data,
                        )
                        before_bytes = self.result_cache.get_memory_total_memory_usage()
                        self._manage_result_cache(node_id, transformed_modality)
                        after_bytes = self.result_cache.get_memory_total_memory_usage()
                        self.scheduler.update_cpu_memory_in_use(
                            after_bytes - before_bytes
                        )
                        self.scheduler.complete_node(node_id)

                    submit_new_ready_nodes()
        assert len(self.result_cache.ref_count.keys()) == 0

        self.result_cache.cleanup_all()
        self._cleanup_leaf_shared_memory()
        return list(task_results.values())

    def _materialize_leaf_modalities_in_shared_memory(self):
        self._leaf_shm_names = []
        for modality in self.modalities:
            if hasattr(modality, "extract_raw_data") and not modality.has_data():
                modality.extract_raw_data()
            data, shm_name, _, _ = add_shared_memory_candidate(modality.data)
            if shm_name is not None:
                modality.data = data
                self._leaf_shm_names.append(shm_name)

    def _cleanup_leaf_shared_memory(self):
        for shm_name in getattr(self, "_leaf_shm_names", []):
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        self._leaf_shm_names = []

    def _checkpoint_memory_usage(
        self,
        node_id: str,
        peak_bytes: int,
        gpu_peak_bytes: int,
        operation_name: str,
        data,
        result,
    ):
        self.memory_usage_checkpoint.increment(node_id)

        shape = None
        if DEBUG:
            shape = self._print_node_stats(node_id, result, operation_name)
            if peak_bytes > self.scheduler.node_resources[node_id][0]:
                print(
                    f"UNDERESTIMATED PEAK MEMORY: Peak bytes: {peak_bytes/1024**3:.2f} GB, Estimated CPU bytes: {self.scheduler.node_resources[node_id][0]/1024**3:.2f} GB for node {node_id}: {operation_name}"
                )
            if gpu_peak_bytes > self.scheduler.node_resources[node_id][1]:
                print(
                    f"UNDERESTIMATED GPU PEAK MEMORY: GPU peak bytes: {gpu_peak_bytes/1024**3:.2f} GB, Estimated GPU bytes: {self.scheduler.node_resources[node_id][1]/1024**3:.2f} GB for node {node_id}: {operation_name}"
                )
            if self.scheduler.node_resources[node_id][0] >= peak_bytes * 2:
                print(
                    f"Peak bytes: {peak_bytes/1024**3:.2f} GB, Estimated CPU bytes: {self.scheduler.node_resources[node_id][0]/1024**3:.2f} GB, 200% of estimated for node {node_id}: {operation_name}"
                )
            if self.scheduler.node_resources[node_id][1] > gpu_peak_bytes * 2:
                print(
                    f"GPU peak bytes: {gpu_peak_bytes/1024**3:.2f} GB, Estimated GPU bytes: {self.scheduler.node_resources[node_id][1]/1024**3:.2f} GB, 200% of estimated for node {node_id}: {operation_name}"
                )
        data[node_id] = {
            "cpu_peak_bytes": peak_bytes,
            "gpu_peak_bytes": gpu_peak_bytes,
            "operation_name": operation_name,
            "estimated_cpu_bytes": self.scheduler.node_resources[node_id][0],
            "estimated_gpu_bytes": self.scheduler.node_resources[node_id][1],
            "shape": shape,
        }
        self.memory_usage_checkpoint.checkpoint_if_due(data)

    def _print_node_stats(self, node_id: str, result: Any, operation_name: str):
        if (
            result is not None
            and operation_name != "BoW"
            and not operation_name.endswith("Split")
        ):
            node_stats = self.scheduler.node_stats[node_id]
            shape = None
            if isinstance(result[0], list):
                if isinstance(result[0][0], np.ndarray):
                    shape = (len(result[0]), *result[0][0].shape)
                elif isinstance(result[0][0], list):
                    shape = (len(result[0]), *result[0][0][0].shape)
                else:
                    shape = (len(result[0]), *result[0][0].shape)
            else:
                shape = result[0].shape
            print(
                f"Node {node_id} {operation_name} should have shape of {node_stats.num_instances, node_stats.output_shape}, actual shape: {len(result), shape} output shape is known: {node_stats.output_shape_is_known}"
            )
            if node_stats.output_shape_is_known:
                assert (
                    len(result) == node_stats.num_instances
                ), f"Node {node_id} {operation_name} should have {node_stats.num_instances} instances, actual: {len(result)}"
            return shape

    def _infer_actual_output_stats(
        self, transformed_modality: Any
    ) -> Optional[RepresentationStats]:
        if transformed_modality is None or not hasattr(transformed_modality, "data"):
            return None

        data = transformed_modality.data

        if isinstance(data, np.ndarray):
            if data.ndim == 0:
                return RepresentationStats(1, (1,), output_shape_is_known=True)
            num_instances = int(data.shape[0])
            output_shape = (
                tuple(int(d) for d in data.shape[1:]) if data.ndim > 1 else (1,)
            )
            return RepresentationStats(
                num_instances, output_shape, output_shape_is_known=True
            )

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], np.ndarray):
            num_instances = len(data)
            first_shape = tuple(int(d) for d in data[0].shape)
            same_shape = all(
                isinstance(x, np.ndarray) and x.shape == data[0].shape for x in data
            )
            return RepresentationStats(
                num_instances,
                first_shape,
                output_shape_is_known=bool(same_shape),
            )

        return None

    def _manage_result_cache(self, node_id: str, result: Any):
        parent_node_id = self.scheduler.get_valid_parent(node_id)
        if parent_node_id is not None:
            self.result_cache.dec_ref(parent_node_id)

        if self.scheduler.get_children(node_id):
            for _ in self.scheduler.get_children(node_id):
                self.result_cache.inc_ref(node_id)
            self.result_cache.add_result(node_id, result)

        if (
            parent_node_id in self.result_cache.ref_count
            and self.result_cache.ref_count[parent_node_id] == 0
        ):
            self.result_cache.clear(parent_node_id)

    def _get_nodes_by_ids(self, nodes_ids: List[str]) -> List[RepresentationNode]:
        return [self.scheduler.mapping[node_id] for node_id in nodes_ids]

    def _get_dag_from_node_ids(self, node_id: str) -> RepresentationDag:
        for dag in self.dags:
            if dag.root_node_id == node_id:
                return dag
        return None

    @staticmethod
    def _is_task_node(node: RepresentationNode) -> bool:
        return bool(getattr(node, "parameters", {}).get("_node_kind") == "task")


@dataclass
class ResultEntry:
    val_score: PerformanceMeasure = None
    train_score: PerformanceMeasure = None
    test_score: PerformanceMeasure = None
    representation_time: float = 0.0
    task_time: float = 0.0
    dag: RepresentationDag = None
    tradeoff_score: float = 0.0
