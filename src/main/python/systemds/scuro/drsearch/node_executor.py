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
from systemds.scuro import Modality
from systemds.scuro.drsearch.node_scheduler import MemoryAwareNodeScheduler
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationNode,
)
import resource
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
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.utils.checkpointing import CheckpointManager
import sys


class RefCountResultCache:
    def __init__(self):
        self.cache = {}
        self.ref_count = {}
        self.memory_usage_per_node = {}

    def get(self, node_id: str) -> Any:
        return self.cache[node_id]

    def add_result(self, node_id: str, result: Any):
        self.cache[node_id] = result
        self.memory_usage_per_node[node_id] = result.calculate_memory_usage()
        print(
            f"Node {node_id} has a CPU memory usage of {self.memory_usage_per_node[node_id]/1024**3:.5f} GB"
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

    def clear(self, node_id: str):
        del self.cache[node_id]
        del self.ref_count[node_id]

    def __len__(self):
        return len(self.cache)

    def get_memory_total_memory_usage(self):
        return sum(self.memory_usage_per_node.values())


def _execute_node_worker(
    node: RepresentationNode,
    input_mods: List[Any],
    task: Any,
    rep_cache: Optional[Dict[str, Any]],
    gpu_id: Optional[int],
):
    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss  # bytes

    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    result = None
    node_operation = node.operation(params=node.parameters)
    operation_name = node_operation.name
    # print(
    #     f"Executing node {node.node_id} inputs: {input_mods[0].modality_id}, gpu: {gpu_id}, operation: {operation_name}"
    # )
    if gpu_id is not None and hasattr(node_operation, "gpu_id"):
        node_operation.gpu_id = gpu_id

    if len(input_mods) == 1:
        if isinstance(node_operation, Context):
            result = input_mods[0].context(node_operation)
        elif isinstance(node_operation, DimensionalityReduction):
            result = input_mods[0].dimensionality_reduction(node_operation)
        elif isinstance(node_operation, AggregatedRepresentation):
            result = node_operation.transform(input_mods[0])
        elif isinstance(node_operation, UnimodalRepresentation):
            if rep_cache is not None and node_operation.name in rep_cache:
                result = rep_cache[node_operation.name]
            else:
                result = input_mods[0].apply_representation(node_operation)
        else:
            result = input_mods[0].apply_representation(node_operation)
    else:
        fusion_op = node_operation
        if hasattr(fusion_op, "needs_training") and fusion_op.needs_training:
            result = input_mods[0].combine_with_training(
                input_mods[1:], fusion_op, task
            )
        else:
            result = input_mods[0].combine(input_mods[1:], fusion_op)
    delta_bytes = proc.memory_info().rss - before
    gpu_peak_bytes = (
        torch.cuda.max_memory_allocated(device) if gpu_id is not None else 0
    )
    # print(f"Node {node.node_id}: {operation_name} has a CPU peak memory usage of {delta_bytes/1024**3:.2f} GB, and a GPU peak memory usage of {gpu_peak_bytes/1024**3:.2f} GB")
    return {
        "result": result,
        "peak_bytes": delta_bytes,
        "gpu_peak_bytes": gpu_peak_bytes,
        "operation_name": operation_name,
    }


def _execute_task_worker(
    task_node_id: str, task: Any, data: Any, gpu_id: Optional[int]
) -> Dict[str, Any]:
    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss  # bytes

    # print(f"Executing task {task_node_id} on GPU {gpu_id}")
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    if gpu_id is not None and hasattr(task, "model") and hasattr(task.model, "device"):
        task.model.device = torch.device(f"cuda:{gpu_id}")
    start = time.perf_counter()
    scores = task.run(data)
    end = time.perf_counter()
    delta_bytes = proc.memory_info().rss - before
    gpu_peak_bytes = (
        torch.cuda.max_memory_allocated(device) if gpu_id is not None else 0
    )
    # print(f"Task {task_node_id} has a CPU peak memory usage of {delta_bytes/1024**3:.2f} GB, and a GPU peak memory usage of {gpu_peak_bytes/1024**3:.2f} GB")
    return {
        "scores": scores,
        "task_time": end - start,
        "peak_bytes": delta_bytes,
        "gpu_peak_bytes": gpu_peak_bytes,
    }


# TODO: checkpoint for memory estimates, name of operation and node id plus estimated and actual memory usage


class NodeExecutor:
    def __init__(
        self,
        dags: List[RepresentationDag],
        modalities: List[Modality],
        tasks: List[Any],
        checkpoint_manager: Optional[CheckpointManager] = None,
        max_num_workers: int = -1,
    ):
        available_total_cpu = float(psutil.virtual_memory().available)
        self.dags = dags
        self.scheduler = MemoryAwareNodeScheduler(
            dags, modalities, tasks, available_total_cpu
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.getcwd(),
            prefix="node_executor_checkpoint_",
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
            checkpoint_dir=os.getcwd(),
            prefix="memory_usage_checkpoint_",
            checkpoint_every=1,
            resume=False,
        )

    def run(self) -> None:
        task_results = {}
        memory_usage_data = {}
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
                        )
                        self.scheduler.complete_node(node_id)

                    else:
                        transformed_modality = result["result"]
                        self._checkpoint_memory_usage(
                            node_id,
                            peak_bytes,
                            gpu_peak_bytes,
                            result["operation_name"],
                            memory_usage_data,
                        )
                        before_bytes = self.result_cache.get_memory_total_memory_usage()
                        self._manage_result_cache(node_id, transformed_modality)
                        after_bytes = self.result_cache.get_memory_total_memory_usage()
                        self.scheduler.update_cpu_memory_in_use(
                            after_bytes - before_bytes
                        )
                        self.scheduler.complete_node(node_id)

                    submit_new_ready_nodes()

        return list(task_results.values())

    def _checkpoint_memory_usage(
        self,
        node_id: str,
        peak_bytes: int,
        gpu_peak_bytes: int,
        operation_name: str,
        data,
    ):
        self.memory_usage_checkpoint.increment(node_id)
        data[node_id] = {
            "cpu_peak_bytes": peak_bytes,
            "gpu_peak_bytes": gpu_peak_bytes,
            "operation_name": operation_name,
            "estimated_cpu_bytes": self.scheduler.node_resources[node_id][0],
            "estimated_gpu_bytes": self.scheduler.node_resources[node_id][1],
        }
        print(
            f"Node {node_id}: {operation_name} has a CPU peak memory usage of {peak_bytes/1024**3:.2f}/{self.scheduler.node_resources[node_id][0]/1024**3:.2f} GB estimated, and a GPU peak memory usage of {gpu_peak_bytes/1024**3:.2f}/{self.scheduler.node_resources[node_id][1]/1024**3:.2f} GB estimated "
        )
        self.memory_usage_checkpoint.checkpoint_if_due(data)

    def _result_cache_size_bytes(self) -> int:
        size = 0
        for node_id in self.result_cache.cache:
            if isinstance(self.result_cache.cache[node_id].data, np.ndarray):
                size += self.result_cache.cache[node_id].data.nbytes
            elif isinstance(self.result_cache.cache[node_id].data, list):
                for item in self.result_cache.cache[node_id].data:
                    if isinstance(item, np.ndarray):
                        size += item.nbytes
                    elif isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, np.ndarray):
                                size += sub.nbytes
            else:
                size += sys.getsizeof(self.result_cache.cache[node_id].data)
        return size

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
