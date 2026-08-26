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
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from systemds.scuro import Modality
from systemds.scuro.drsearch.modality_result_cache import RefCountResultCache
from systemds.scuro.drsearch.modality_shared_memory import (
    add_shared_memory_candidate,
    collect_shm_names_from_payload,
    unlink_shm,
)
from systemds.scuro.drsearch.node_scheduler import MemoryAwareNodeScheduler
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationNode,
)
from systemds.scuro.drsearch.task import PerformanceMeasure
from systemds.scuro.drsearch.worker_pool import PersistentWorkerPool, create_mp_context
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.context import Context
from systemds.scuro.representations.dimensionality_reduction import (
    DimensionalityReduction,
)
from systemds.scuro.representations.representation import (
    RepresentationStats,
    infer_stats_from_data,
)
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.utils.checkpointing import CheckpointManager
from systemds.scuro.utils.memory_utility import (
    MemoryMeasurement,
    cleanup_gpu,
    cpu_memory_budget_bytes,
    estimate_modality_bytes,
    is_cuda_oom,
    measure_memory_during,
)
from systemds.scuro.utils.static_variables import DEBUG

_MAX_NODE_RETRIES = int(os.environ.get("SCURO_MAX_NODE_RETRIES", "3"))


def _run_gpu_op(fn, gpu_id: Optional[int]):
    if gpu_id is None or not torch.cuda.is_available():
        return fn()
    try:
        try:
            return fn()
        except Exception as e:
            if is_cuda_oom(e):
                cleanup_gpu(gpu_id)
                _WORKER_OP_CACHE.clear()
                return fn()
            raise
    finally:
        cleanup_gpu(gpu_id)


_WORKER_OP_CACHE: Dict[str, Any] = {}


def _instantiate_operation(node):
    cache_key = None
    if getattr(node.operation, "cache_in_worker", False):
        try:
            params_repr = repr(sorted(node.parameters.items(), key=lambda kv: kv[0]))
            cache_key = (
                f"{node.operation.__module__}.{node.operation.__qualname__}"
                f"|{params_repr}"
            )
        except Exception:
            cache_key = None
    if cache_key is not None and cache_key in _WORKER_OP_CACHE:
        return _WORKER_OP_CACHE[cache_key]
    operation = node.operation(params=node.parameters)
    if cache_key is not None:
        _WORKER_OP_CACHE[cache_key] = operation
    return operation


def _infer_actual_output_stats(
    transformed_modality: Any,
) -> Optional[RepresentationStats]:
    if transformed_modality is None or not hasattr(transformed_modality, "data"):
        return None
    return infer_stats_from_data(transformed_modality.data)


def _offload_to_shared_memory(result: Any):
    if result is None or not hasattr(result, "data"):
        return None, None, 0, None

    actual_stats = _infer_actual_output_stats(result)
    shm_name = None
    resident_bytes = None
    shm_bytes = 0
    try:
        resident_bytes = result.calculate_memory_usage()
        data, shm_name, shm_bytes, resident_bytes = add_shared_memory_candidate(
            result.data, resident_bytes
        )
        if data is not None:
            result._data = data
    except Exception as e:
        shm_name = None
        shm_bytes = 0
        print(f"Failed to move worker result to shared memory: {e}")

    return shm_name, resident_bytes, int(shm_bytes or 0), actual_stats


def _execute_node_worker(node, input_mods: List[Any], gpu_id: Optional[int]):
    start_time = time.perf_counter()
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    node_operation = _instantiate_operation(node)
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
                agg = (
                    AggregatedRepresentation(params=pushdown_config)
                    if pushdown_config is not None
                    else None
                )
                return input_mods[0].apply_representation(
                    node_operation, aggregation=agg
                )
            return input_mods[0].apply_representation(node_operation)
        else:
            fusion_op = node_operation
            if getattr(fusion_op, "needs_training", False):
                return input_mods[0].combine_with_training(
                    input_mods[1:], fusion_op, None
                )
            return input_mods[0].combine(input_mods[1:], fusion_op)

    gpu_peak_bytes = -1
    measurement = None
    if DEBUG:
        input_resident = sum(estimate_modality_bytes(m) for m in input_mods)
        result, measurement = measure_memory_during(
            lambda: _run_gpu_op(_run_node_op, gpu_id),
            input_resident_bytes=input_resident,
            sample_s=0.01,
        )
        gpu_peak_bytes = (
            torch.cuda.max_memory_allocated(device) if gpu_id is not None else 0
        )
    else:
        result = _run_gpu_op(_run_node_op, gpu_id)

    shm_name, resident_bytes, shm_bytes, actual_stats = _offload_to_shared_memory(
        result
    )
    end_time = time.perf_counter()
    return {
        "result": result,
        "result_shm_name": shm_name,
        "result_resident_bytes": resident_bytes,
        "result_shm_bytes": shm_bytes,
        "actual_stats": actual_stats,
        "memory": measurement,
        "peak_bytes": measurement.increment_bytes if measurement else -1,
        "gpu_peak_bytes": gpu_peak_bytes,
        "operation_name": operation_name,
        "start_time": start_time,
        "end_time": end_time,
        "pid": os.getpid(),
    }


def _execute_task_worker(
    task_node_id: str,
    task: Any,
    modality: Any,
    gpu_id: Optional[int],
    aggregation=None,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
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
        if aggregation is not None:
            data = (
                aggregation.operation(params=aggregation.parameters)
                .transform(modality)
                .data
            )
        else:
            data = modality.data
        scores = task.run(data)
        end = time.perf_counter()
        return scores, end - start

    gpu_peak_bytes = -1
    measurement = None
    if DEBUG:
        result, measurement = measure_memory_during(
            lambda: _run_gpu_op(_run_task, gpu_id),
            input_resident_bytes=estimate_modality_bytes(modality),
            sample_s=0.01,
        )
        gpu_peak_bytes = (
            torch.cuda.max_memory_allocated(device) if gpu_id is not None else 0
        )
    else:
        result = _run_gpu_op(_run_task, gpu_id)
    end_time = time.perf_counter()
    return {
        "scores": result[0],
        "task_time": result[1],
        "memory": measurement,
        "peak_bytes": measurement.increment_bytes if measurement else -1,
        "gpu_peak_bytes": gpu_peak_bytes,
        "start_time": start_time,
        "end_time": end_time,
        "pid": os.getpid(),
    }


def _execute_leaf_batch_worker(nodes: List[Any], modality: Any, gpu_id: Optional[int]):
    node_id_by_representation = {}

    def _run():
        representations = []
        for node in nodes:
            operation = node.operation(params=node.parameters)
            if hasattr(operation, "gpu_id"):
                operation.gpu_id = gpu_id
            representations.append(operation)
            node_id_by_representation[operation.name] = node.node_id
        return modality.apply_representations(representations, parallel=True)

    modality_results = _run_gpu_op(_run, gpu_id)
    shm_info = {}
    for representation_name, transformed_modality in modality_results.items():
        shm_name, resident_bytes, shm_bytes, actual_stats = _offload_to_shared_memory(
            transformed_modality
        )
        shm_info[representation_name] = {
            "shm_name": shm_name,
            "resident_bytes": resident_bytes,
            "shm_bytes": shm_bytes,
            "actual_stats": actual_stats,
        }
    return {
        "results": modality_results,
        "node_id_by_representation": node_id_by_representation,
        "shm_info": shm_info,
    }


def _load_leaf_worker(modality: Any) -> Dict[str, Any]:
    if hasattr(modality, "extract_raw_data") and not modality.has_data():
        modality.extract_raw_data()

    data = modality.data
    resident_bytes = 0
    try:
        resident_bytes = modality.estimate_memory_bytes()
    except Exception:
        resident_bytes = 0

    wrapped, shm_name, _, resident_bytes = add_shared_memory_candidate(
        data, resident_bytes
    )
    if wrapped is not None:
        data = wrapped

    return {"data": data, "metadata": modality.metadata, "shm_name": shm_name}


def _dispatch_node(payload, gpu_id):
    node, input_mods = payload
    return _execute_node_worker(node, input_mods, gpu_id)


def _dispatch_task(payload, gpu_id):
    task_node_id, task, modality, aggregation = payload
    return _execute_task_worker(task_node_id, task, modality, gpu_id, aggregation)


def _dispatch_leaf_batch(payload, gpu_id):
    nodes, modality = payload
    return _execute_leaf_batch_worker(nodes, modality, gpu_id)


def _dispatch_load_leaf(payload, _gpu_id):
    (modality,) = payload
    return _load_leaf_worker(modality)


_WORKER_DISPATCH = {
    "node": _dispatch_node,
    "task": _dispatch_task,
    "leaf_batch": _dispatch_leaf_batch,
    "load_leaf": _dispatch_load_leaf,
}


@dataclass
class _NodeUnit:
    node_id: str


@dataclass
class _BatchUnit:
    node_ids: List[str]


@dataclass
class ResultEntry:
    val_score: PerformanceMeasure = None
    train_score: PerformanceMeasure = None
    test_score: PerformanceMeasure = None
    representation_time: float = 0.0
    task_time: float = 0.0
    dag: RepresentationDag = None
    tradeoff_score: float = 0.0


class NodeExecutor:
    def __init__(
        self,
        dags: List[RepresentationDag],
        modalities: List[Modality],
        tasks: List[Any],
        max_num_workers: int = -1,
        result_path: Optional[str] = None,
        enable_checkpointing: bool = False,
        worker_pool: Optional[PersistentWorkerPool] = None,
        search_start: Optional[float] = None,
    ):
        self.enable_checkpointing = enable_checkpointing
        available_total_cpu = cpu_memory_budget_bytes()
        self.dags = dags
        self.scheduler = MemoryAwareNodeScheduler(
            dags, modalities, tasks, available_total_cpu
        )
        self._checkpoint_manager = CheckpointManager(
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
        self._modalities = modalities
        self._tasks = tasks
        self._result_path = result_path
        self._result_cache = RefCountResultCache()
        self._memory_usage_checkpoint = CheckpointManager(
            checkpoint_dir=result_path if result_path is not None else os.getcwd(),
            prefix=f"memory_usage_checkpoint_{modalities[0].modality_id}_",
            checkpoint_every=1,
            resume=False,
        )
        self._memory_usage_data: Dict[str, Any] = {}
        self.statistics = {"worker_stats": {}, "node_stats": {}}
        self._eval_counter = 0
        self._nodes_executed = 0
        self._search_start = (
            search_start if search_start is not None else time.perf_counter()
        )

        self._node_attempts: Dict[str, int] = {}

        self._job_units: Dict[int, Union[_NodeUnit, _BatchUnit]] = {}
        self._job_retained_shm: Dict[int, List[str]] = {}
        self._leaf_shm_names: List[str] = []
        self._task_results: Dict[str, ResultEntry] = {}

        self._owns_pool = worker_pool is None
        if worker_pool is None:
            cpu_count = os.cpu_count() or 1
            threads_per_worker = max(1, cpu_count // max(1, self.max_num_workers))
            worker_pool = PersistentWorkerPool(
                self.max_num_workers,
                _WORKER_DISPATCH,
                ctx=create_mp_context(),
                threads_per_worker=threads_per_worker,
            )
        self._pool = worker_pool

    def _requeue_or_give_up(self, node_id: str, reason: str) -> bool:
        attempts = self._node_attempts.get(node_id, 0) + 1
        self._node_attempts[node_id] = attempts
        if attempts > _MAX_NODE_RETRIES:
            print(
                f"[node_executor] giving up on node {node_id} after {attempts} "
                f"failed attempts ({reason}); marking it permanently failed and "
                f"continuing with the rest of the search.",
                flush=True,
            )
            self.scheduler.add_failed_node(node_id, reason)
            return False
        print(
            f"[node_executor] node {node_id} did not complete (attempt "
            f"{attempts}/{_MAX_NODE_RETRIES + 1}): {reason}. Re-queuing it for "
            f"another attempt.",
            flush=True,
        )
        self.scheduler.requeue_node(node_id)
        return True

    def _release_parents(self, node_id: str) -> None:
        for parent_id in self.scheduler.get_valid_parents(node_id):
            self._result_cache.dec_ref(parent_id)

    def _retain_for_submit(self, parent_ids: List[str], payload: Any) -> List[str]:
        names: List[str] = []
        for parent_id in parent_ids or []:
            names.extend(self._result_cache.shared_memory_names.get(parent_id, []))
        if not parent_ids:
            names.extend(self._leaf_shm_names)
        names.extend(collect_shm_names_from_payload(payload))
        names = list(dict.fromkeys(names))
        return self._result_cache.retain_shm_names(names)

    def _load_leaf_modalities(self) -> None:
        for modality in self._modalities:
            if getattr(modality, "has_data", None) and modality.has_data():
                continue
            attempts = 0
            while True:
                self._pool.submit("load_leaf", (modality,), gpu_id=None)
                jr = self._pool.wait()
                if jr.ok:
                    break
                attempts += 1
                if attempts > _MAX_NODE_RETRIES:
                    raise RuntimeError(
                        f"Failed to load leaf modality {modality.modality_id}: "
                        f"{jr.error}"
                    )
            modality._data = jr.value["data"]
            modality.metadata = jr.value["metadata"]
            shm_name = jr.value.get("shm_name")
            if shm_name is not None:
                self._leaf_shm_names.append(shm_name)

    def _cleanup_leaf_shared_memory(self) -> None:
        for shm_name in self._leaf_shm_names:
            unlink_shm(shm_name)
        self._leaf_shm_names = []

    def _submit_node(self, node_id: str) -> None:
        node = self.scheduler.mapping[node_id]
        gpu_id = node.gpu_id
        parent_ids = self.scheduler.get_valid_parents(node_id)
        parent_results = (
            [self._result_cache.get(pid) for pid in parent_ids] if parent_ids else None
        )

        if self._is_task_node(node):
            task_idx = int(node.parameters.get("_task_idx", 0))
            payload = (
                self._modalities[0] if parent_results is None else parent_results[0]
            )
            self._task_results[node_id] = ResultEntry(
                dag=self._get_dag_from_node_ids(node_id),
                representation_time=payload.transform_time,
            )
            retained = self._retain_for_submit(parent_ids, payload)
            self.scheduler.begin_execution(node_id)
            self.scheduler.move_to_running(node_id)
            job_id = self._pool.submit(
                "task",
                (node_id, self._tasks[task_idx], payload, node.aggregation),
                gpu_id=gpu_id,
            )
        else:
            payload = self._modalities if parent_results is None else parent_results
            retained = self._retain_for_submit(parent_ids, payload)
            self.scheduler.begin_execution(node_id)
            self.scheduler.move_to_running(node_id)
            job_id = self._pool.submit("node", (node, payload), gpu_id=gpu_id)

        self._job_units[job_id] = _NodeUnit(node_id)
        self._job_retained_shm[job_id] = retained

    def _submit_leaf_batch(self, node_ids: List[str]) -> None:
        nodes = [self.scheduler.mapping[nid] for nid in node_ids]
        gpu_id = nodes[0].gpu_id
        retained = self._retain_for_submit([], self._modalities[0].data)
        for nid in node_ids:
            self.scheduler.begin_execution(nid)
        self.scheduler.move_to_running(node_ids)
        job_id = self._pool.submit(
            "leaf_batch", (nodes, self._modalities[0]), gpu_id=gpu_id
        )
        self._job_units[job_id] = _BatchUnit(node_ids)
        self._job_retained_shm[job_id] = retained

    def _fill_pipeline(self) -> None:
        ready = self.scheduler.get_runnable().copy()
        for entry in ready:
            if not self._pool.has_idle_worker:
                break
            if isinstance(entry, list):
                self._submit_leaf_batch(entry)
            else:
                if not self.scheduler.can_start_now(entry):
                    continue
                self._submit_node(entry)

    def _record_stats(self, node_id: str, pid: int, start_time: float, end_time: float):
        node_stats = self.statistics["node_stats"]
        worker_stats = self.statistics["worker_stats"]
        node_stats[node_id] = {"start_time": start_time, "end_time": end_time}
        entry = worker_stats.get(pid)
        if entry is None:
            worker_stats[pid] = {
                "start_time": start_time,
                "end_time": end_time,
                "busy_time": end_time - start_time,
                "num_jobs": 1,
            }
        else:
            entry["start_time"] = min(entry["start_time"], start_time)
            entry["end_time"] = max(entry["end_time"], end_time)
            entry["busy_time"] += end_time - start_time
            entry["num_jobs"] += 1

    def _process_result(self, jr) -> None:
        unit = self._job_units.pop(jr.job_id, None)
        retained = self._job_retained_shm.pop(jr.job_id, [])
        try:
            if unit is None:
                return
            if not jr.ok:
                self._handle_job_failure(unit, jr)
                return
            if isinstance(unit, _BatchUnit):
                self._handle_batch_success(jr.value)
            else:
                self._handle_node_success(unit.node_id, jr.value)
        finally:
            if retained:
                self._result_cache.release_shm_names(retained)

    def _handle_job_failure(self, unit: Union[_NodeUnit, _BatchUnit], jr):
        reason = jr.error or "unknown worker failure"
        if jr.cuda_oom:
            reason += " (CUDA out of memory)"
        node_ids = unit.node_ids if isinstance(unit, _BatchUnit) else [unit.node_id]
        for node_id in node_ids:
            requeued = self._requeue_or_give_up(node_id, reason)
            if not requeued:
                self._release_parents(node_id)

    def _handle_node_success(self, node_id: str, value: Dict[str, Any]) -> None:
        node = self.scheduler.mapping[node_id]
        if "pid" in value:
            self._record_stats(
                node_id, value["pid"], value["start_time"], value["end_time"]
            )

        if self._is_task_node(node):
            entry = self._task_results[node_id]
            entry.task_time = value["task_time"]
            entry.train_score = value["scores"][0].average_scores
            entry.val_score = value["scores"][1].average_scores
            entry.test_score = value["scores"][2].average_scores
            if self.enable_checkpointing:
                self._checkpoint_manager.increment(node_id)
                self._checkpoint_manager.checkpoint_if_due(
                    self._task_results, self._discard_report()
                )
                self._checkpoint_memory_usage(
                    node_id,
                    value["peak_bytes"],
                    value["gpu_peak_bytes"],
                    "task",
                    None,
                    measurement=value.get("memory"),
                )
            self._release_parents(node_id)
            self.scheduler.complete_node(node_id)
        else:
            self._handle_modality_result(
                value["result"],
                node_id,
                value["peak_bytes"],
                value["gpu_peak_bytes"],
                value["operation_name"],
                actual_stats=value.get("actual_stats"),
                shm_name=value.get("result_shm_name"),
                resident_bytes=value.get("result_resident_bytes"),
                shm_bytes=value.get("result_shm_bytes", 0),
                measurement=value.get("memory"),
            )

    def _handle_batch_success(self, value: Dict[str, Any]) -> None:
        results = value["results"]
        node_id_by_representation = value["node_id_by_representation"]
        shm_info = value.get("shm_info", {})
        for representation, transformed_modality in results.items():
            node_id = node_id_by_representation[representation]
            info = shm_info.get(representation, {})
            self._handle_modality_result(
                transformed_modality,
                node_id,
                None,
                None,
                representation,
                actual_stats=info.get("actual_stats"),
                shm_name=info.get("shm_name"),
                resident_bytes=info.get("resident_bytes"),
                shm_bytes=info.get("shm_bytes", 0),
            )

    def _handle_modality_result(
        self,
        transformed_modality: Any,
        node_id: str,
        peak_bytes: Optional[int],
        gpu_peak_bytes: Optional[int],
        operation_name: str,
        actual_stats: Optional[RepresentationStats] = None,
        shm_name: Optional[str] = None,
        resident_bytes: Optional[int] = None,
        shm_bytes: int = 0,
        measurement: Optional[MemoryMeasurement] = None,
    ):
        if actual_stats is None:
            actual_stats = _infer_actual_output_stats(transformed_modality)
        estimated_stats = self.scheduler.node_stats.get(node_id)

        if actual_stats is not None and (
            estimated_stats is None
            or not getattr(estimated_stats, "output_shape_is_known", True)
        ):
            self.scheduler.update_node_stats_and_reestimate_descendants(
                node_id, actual_stats
            )
        if self.enable_checkpointing:
            self._checkpoint_memory_usage(
                node_id,
                peak_bytes,
                gpu_peak_bytes,
                operation_name,
                actual_stats,
                measurement=measurement,
            )
        before_bytes = self._result_cache.get_memory_total_memory_usage()
        self._manage_result_cache(
            node_id,
            transformed_modality,
            shm_name=shm_name,
            resident_bytes=resident_bytes,
            shm_bytes=shm_bytes,
        )
        after_bytes = self._result_cache.get_memory_total_memory_usage()
        self.scheduler.update_cpu_memory_in_use(after_bytes - before_bytes)
        self.scheduler.complete_node(node_id)

    def _manage_result_cache(
        self,
        node_id: str,
        result: Any,
        shm_name: Optional[str] = None,
        resident_bytes: Optional[int] = None,
        shm_bytes: int = 0,
    ):
        self._release_parents(node_id)

        children = self.scheduler.get_children(node_id)
        if children:
            for _ in children:
                self._result_cache.inc_ref(node_id)
            self._result_cache.add_result(
                node_id,
                result,
                shm_name=shm_name,
                resident_bytes=resident_bytes,
                shm_bytes=shm_bytes,
            )
        elif shm_name is not None:
            unlink_shm(shm_name)

    def _checkpoint_memory_usage(
        self,
        node_id: str,
        peak_bytes: Optional[int],
        gpu_peak_bytes: Optional[int],
        operation_name: str,
        actual_stats: Optional[RepresentationStats],
        measurement: Optional[MemoryMeasurement] = None,
    ):
        if self.enable_checkpointing:
            self._memory_usage_checkpoint.increment(node_id)

        if measurement is not None:
            peak_bytes = measurement.footprint_bytes

        shape = None
        if DEBUG:
            shape = self._print_node_stats(node_id, actual_stats, operation_name)
            est_cpu, est_gpu = self.scheduler.node_resources[node_id]
            if peak_bytes is not None and peak_bytes >= 0:
                if peak_bytes > est_cpu:
                    print(
                        f"UNDERESTIMATED PEAK MEMORY: Peak bytes: {peak_bytes/1024**3:.2f} GB, "
                        f"Estimated CPU bytes: {est_cpu/1024**3:.2f} GB for node {node_id}: {operation_name}"
                    )
                if est_cpu >= peak_bytes * 2:
                    print(
                        f"Peak bytes: {peak_bytes/1024**3:.2f} GB, Estimated CPU bytes: "
                        f"{est_cpu/1024**3:.2f} GB, >200% of estimated for node {node_id}: {operation_name}"
                    )
            if gpu_peak_bytes is not None and gpu_peak_bytes >= 0:
                if gpu_peak_bytes > est_gpu:
                    print(
                        f"UNDERESTIMATED GPU PEAK MEMORY: GPU peak bytes: {gpu_peak_bytes/1024**3:.2f} GB, "
                        f"Estimated GPU bytes: {est_gpu/1024**3:.2f} GB for node {node_id}: {operation_name}"
                    )
                if est_gpu > gpu_peak_bytes * 2:
                    print(
                        f"GPU peak bytes: {gpu_peak_bytes/1024**3:.2f} GB, Estimated GPU bytes: "
                        f"{est_gpu/1024**3:.2f} GB, >200% of estimated for node {node_id}: {operation_name}"
                    )
        if self.enable_checkpointing:
            self._memory_usage_data[node_id] = {
                "cpu_peak_bytes": peak_bytes if peak_bytes is not None else -1,
                "gpu_peak_bytes": gpu_peak_bytes if gpu_peak_bytes is not None else -1,
                "operation_name": operation_name,
                "estimated_cpu_bytes": self.scheduler.node_resources[node_id][0],
                "estimated_gpu_bytes": self.scheduler.node_resources[node_id][1],
                "shape": shape,
                "cpu_increment_bytes": (
                    measurement.increment_bytes if measurement else -1
                ),
                "cpu_footprint_bytes": (
                    measurement.footprint_bytes if measurement else -1
                ),
                "input_resident_bytes": (
                    measurement.input_resident_bytes if measurement else -1
                ),
                "traced_peak_bytes": (
                    measurement.traced_peak_bytes if measurement else -1
                ),
                "rss_delta_bytes": measurement.rss_delta_bytes if measurement else -1,
                "num_instances": getattr(actual_stats, "num_instances", None),
                "output_shape": getattr(actual_stats, "output_shape", None),
                "dtype": str(getattr(actual_stats, "dtype", None)),
                "container": str(getattr(actual_stats, "container", None)),
            }
            self._memory_usage_checkpoint.checkpoint_if_due(self._memory_usage_data)

    def _print_node_stats(
        self,
        node_id: str,
        actual_stats: Optional[RepresentationStats],
        operation_name: str,
    ):
        if actual_stats is None:
            return None
        node_stats = self.scheduler.node_stats.get(node_id)
        shape = actual_stats.output_shape
        if node_stats is not None:
            print(
                f"Node {node_id} {operation_name} should have shape of "
                f"{node_stats.num_instances, node_stats.output_shape}, actual shape: "
                f"{actual_stats.num_instances, shape} output shape is known: "
                f"{node_stats.output_shape_is_known}"
            )
        return shape

    def _get_dag_from_node_ids(self, node_id: str) -> Optional[RepresentationDag]:
        for dag in self.dags:
            if dag.root_node_id == node_id:
                return dag
        return None

    def _describe_node(self, node_id: str) -> str:
        names = []
        seen = set()
        current = node_id
        while current is not None and current not in seen:
            seen.add(current)
            node = self.scheduler.mapping.get(current)
            if node is None:
                break
            if node.operation is not None:
                try:
                    names.append(node.operation().name)
                except Exception:
                    names.append(
                        getattr(node.operation, "__name__", str(node.operation))
                    )
            parent_ids = [pid for pid in self.scheduler.get_valid_parents(current)]
            current = parent_ids[0] if len(parent_ids) == 1 else None
            if len(parent_ids) > 1:
                names.append(
                    "["
                    + ", ".join(self._describe_node(pid) for pid in parent_ids)
                    + "]"
                )
        names.reverse()
        return " -> ".join(names) if names else node_id

    def _discard_report(self) -> Dict[str, Any]:
        return {
            "failed_nodes": {
                node_id: {
                    "representation": self._describe_node(node_id),
                    "reason": reason,
                }
                for node_id, reason in self.scheduler.failed_node_reasons.items()
            },
            "blocked_memory_nodes": {
                node_id: {
                    "representation": self._describe_node(node_id),
                    "reason": reason,
                }
                for node_id, reason in self.scheduler.blocked_memory_reasons.items()
            },
            "cpu_fallback_nodes": {
                node_id: {
                    "representation": self._describe_node(node_id),
                    "reason": reason,
                }
                for node_id, reason in self.scheduler.cpu_fallback_reasons.items()
            },
            "deadlock": self.scheduler.deadlock,
            "deadlock_reason": self.scheduler.deadlock_reason,
        }

    @staticmethod
    def _is_task_node(node: RepresentationNode) -> bool:
        return bool(getattr(node, "parameters", {}).get("_node_kind") == "task")

    def run(self) -> Dict[str, Any]:
        self._task_results = {}
        self._load_leaf_modalities()

        try:
            self._fill_pipeline()
            while self._job_units or not self.scheduler.is_finished():
                self._fill_pipeline()
                if not self._job_units:
                    continue
                jr = self._pool.wait()
                self._process_result(jr)
                self._fill_pipeline()
        finally:
            if self._owns_pool:
                self._pool.shutdown()
            self._result_cache.cleanup_all()
            self._cleanup_leaf_shared_memory()

        if self.enable_checkpointing:
            self._checkpoint_manager.save_checkpoint(
                self._task_results, self._discard_report()
            )

        return {
            "task_results": list(self._task_results.values()),
            "statistics": self.statistics,
        }
