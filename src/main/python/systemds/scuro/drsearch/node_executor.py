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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

from systemds.scuro.dataloader.base_loader import LazyFileSequence
from systemds.scuro.modality.modality import Modality
from systemds.scuro.drsearch.modality_result_cache import RefCountResultCache
from systemds.scuro.drsearch.modality_shared_memory import (
    SHARED_MEMORY_MIN_BYTES,
    _should_offload,
    add_shared_memory_candidate,
    collect_shm_names_from_payload,
    move_arrays_to_shared_memory,
    unlink_shm,
)
from systemds.scuro.drsearch.node_scheduler import MemoryAwareNodeScheduler
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationNode,
)
from systemds.scuro.drsearch.task import PerformanceMeasure
from systemds.scuro.drsearch.worker_pool import (
    PersistentWorkerPool,
    create_mp_context,
    default_gpu_slots_per_device,
    default_threads_per_worker,
)
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


def _materialize_lazy_leaves() -> bool:
    return os.environ.get("SCURO_MATERIALIZE_LAZY_LEAVES", "1") == "1"


def _lazy_leaf_budget_fraction() -> float:
    return float(os.environ.get("SCURO_LAZY_LEAF_BUDGET_FRACTION", "0.5"))


def _keep_worker_models() -> bool:
    return os.environ.get("SCURO_KEEP_WORKER_MODELS", "1") == "1"


def _place_on_device(obj: Any, gpu_id: Optional[int]) -> None:
    """Apply scheduler placement even when an object selected CUDA itself."""
    device = torch.device("cpu" if gpu_id is None else f"cuda:{gpu_id}")
    if hasattr(obj, "gpu_id"):
        try:
            obj.gpu_id = gpu_id
        except (AttributeError, RuntimeError):
            pass
    if hasattr(obj, "device"):
        try:
            obj.device = device
        except (AttributeError, RuntimeError):
            pass

    model = obj.__dict__.get("model")
    if model is not None and hasattr(model, "to"):
        try:
            obj.model = model.to(device)
        except (AttributeError, RuntimeError):
            # A read-only "model" means the operator owns its own placement.
            pass


def _run_gpu_op(fn, gpu_id: Optional[int], cleanup_after: bool = True):
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
        if cleanup_after:
            cleanup_gpu(gpu_id)


_WORKER_OP_CACHE: Dict[str, Any] = {}


def _operation_cache_key(node) -> Optional[str]:
    if not getattr(node.operation, "cache_in_worker", False):
        return None
    try:
        params = sorted(
            (kv for kv in node.parameters.items() if not kv[0].startswith("_")),
            key=lambda kv: kv[0],
        )
        return f"{node.operation.__module__}.{node.operation.__qualname__}|{params!r}"
    except Exception:
        return None


def _instantiate_operation(node):
    cache_key = _operation_cache_key(node)
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


def _measure_memory() -> bool:
    return DEBUG or os.environ.get("SCURO_MEASURE_MEMORY", "0") == "1"


def _execute_node_worker(node, input_mods: List[Any], gpu_id: Optional[int]):
    start_time = time.perf_counter()
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    node_operation = _instantiate_operation(node)
    _place_on_device(node_operation, gpu_id)
    operation_name = node_operation.name
    if DEBUG:
        print(f"Executing node {node.node_id} {operation_name} on GPU {gpu_id}")

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
                raise RuntimeError(
                    "Fitted representations require SplitAwareDAGEvaluator"
                )
            return input_mods[0].combine(input_mods[1:], fusion_op)

    gpu_peak_bytes = -1
    measurement = None
    if _measure_memory():
        input_resident = sum(estimate_modality_bytes(m) for m in input_mods)
        result, measurement = measure_memory_during(
            lambda: _run_gpu_op(_run_node_op, gpu_id, cleanup_after=False),
            input_resident_bytes=input_resident,
            sample_s=0.01,
        )
        gpu_peak_bytes = (
            torch.cuda.max_memory_allocated(device) if gpu_id is not None else 0
        )
    else:
        result = _run_gpu_op(_run_node_op, gpu_id, cleanup_after=False)

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

    _place_on_device(getattr(task, "model", task), gpu_id)

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
    if _measure_memory():
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
        "task_timing": getattr(task, "last_run_timing", {}),
        "memory": measurement,
        "peak_bytes": measurement.increment_bytes if measurement else -1,
        "gpu_peak_bytes": gpu_peak_bytes,
        "start_time": start_time,
        "end_time": end_time,
        "pid": os.getpid(),
    }


def _execute_leaf_batch_worker(nodes: List[Any], modality: Any, gpu_id: Optional[int]):
    def _run():
        representations = []
        representation_keys = []
        aggregations = []
        for node in nodes:
            operation = node.operation(params=node.parameters)
            _place_on_device(operation, gpu_id)
            representations.append(operation)
            representation_keys.append(node.node_id)
            pushdown_config = node.parameters.get("_pushdown_aggregation")
            aggregations.append(
                AggregatedRepresentation(params=pushdown_config)
                if pushdown_config is not None
                else None
            )
        return modality.apply_representations(
            representations,
            parallel=True,
            representation_keys=representation_keys,
            aggregations=aggregations,
        )

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
        "shm_info": shm_info,
        "failed_nodes": {
            node_id: f"{type(error).__name__}: {error}"
            for node_id, error in getattr(
                modality, "failed_representations", {}
            ).items()
        },
    }


def _load_leaf_worker(modality: Any, materialize: bool = False) -> Dict[str, Any]:
    if hasattr(modality, "extract_raw_data") and not modality.has_data():
        modality.extract_raw_data()

    data = modality.data
    if materialize and isinstance(data, LazyFileSequence):
        arrays = data.materialize()
        shared, shm_name, shm_bytes = move_arrays_to_shared_memory(arrays)
        if shared is not None:
            return {
                "data": shared,
                "metadata": modality.metadata,
                "shm_name": shm_name,
                "shm_bytes": shm_bytes,
            }
        if sum(getattr(a, "nbytes", 0) for a in arrays) < SHARED_MEMORY_MIN_BYTES:
            return {
                "data": arrays,
                "metadata": modality.metadata,
                "shm_name": None,
                "materialized": True,
            }

    resident_bytes = 0
    try:
        resident_bytes = modality.estimate_peak_memory_bytes()["cpu_peak_bytes"]
    except Exception:
        resident_bytes = 0

    wrapped, shm_name, _, resident_bytes = add_shared_memory_candidate(
        data, resident_bytes
    )
    if wrapped is not None:
        data = wrapped

    return {"data": data, "metadata": modality.metadata, "shm_name": shm_name}


def _retained_gpu_bytes() -> int:
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return 0
    return int(torch.cuda.memory_allocated(0))


def _release_worker_models() -> None:
    _WORKER_OP_CACHE.clear()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cleanup_gpu(0)


def _dispatch_node(payload, gpu_id):
    node, input_mods = payload
    cache_key = _operation_cache_key(node)
    if cache_key is not None and cache_key not in _WORKER_OP_CACHE:
        _release_worker_models()
    keep = cache_key is not None and gpu_id is not None and _keep_worker_models()
    try:
        value = _execute_node_worker(node, input_mods, gpu_id)
    except BaseException:
        keep = False
        raise
    finally:
        if cache_key is not None and not keep:
            _release_worker_models()
        elif gpu_id is not None:
            cleanup_gpu(gpu_id)
    value["cached_operation"] = next(iter(_WORKER_OP_CACHE), None)
    value["retained_gpu_bytes"] = _retained_gpu_bytes()
    return value


def _dispatch_release_models(_payload, _gpu_id):
    _release_worker_models()
    return {"retained_gpu_bytes": _retained_gpu_bytes()}


def _dispatch_task(payload, gpu_id):
    task_node_id, task, modality, aggregation = payload
    return _execute_task_worker(task_node_id, task, modality, gpu_id, aggregation)


def _dispatch_leaf_batch(payload, gpu_id):
    nodes, modality = payload
    return _execute_leaf_batch_worker(nodes, modality, gpu_id)


def _dispatch_load_leaf(payload, _gpu_id):
    modality, materialize = payload
    return _load_leaf_worker(modality, materialize)


_WORKER_DISPATCH = {
    "node": _dispatch_node,
    "task": _dispatch_task,
    "leaf_batch": _dispatch_leaf_batch,
    "load_leaf": _dispatch_load_leaf,
    "release_models": _dispatch_release_models,
}


@dataclass
class _NodeUnit:
    node_id: str


@dataclass
class _ReleaseUnit:
    pid: int


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
    train_fold_scores: dict = field(default_factory=dict)
    val_fold_scores: dict = field(default_factory=dict)
    test_fold_scores: dict = field(default_factory=dict)
    train_score_stds: dict = field(default_factory=dict)
    val_score_stds: dict = field(default_factory=dict)
    test_score_stds: dict = field(default_factory=dict)
    task_timing: dict = field(default_factory=dict)
    peak_cpu_bytes: int = -1
    peak_gpu_bytes: int = -1
    eval_index: int = -1
    t_since_search_start_s: float = 0.0
    t_eval_end_unix: float = 0.0


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
        available_total_cpu = cpu_memory_budget_bytes(
            getattr(getattr(modalities[0], "modality_type", None), "name", None)
            if modalities
            else None
        )
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
        self.statistics = {"worker_stats": {}, "node_stats": {}, "reuse": {}}
        self._cache_consumptions = 0
        self._cache_misses = 0
        self._nodes_executed = 0
        self._peak_cache_bytes = 0
        self._leaf_cached_bytes = self.scheduler.memory_stats["cpu_cached"]
        self._eval_counter = 0
        self._nodes_executed = 0
        self._search_start = (
            search_start if search_start is not None else time.perf_counter()
        )

        self._node_attempts: Dict[str, int] = {}

        self._job_units: Dict[int, Union[_NodeUnit, _BatchUnit, _ReleaseUnit]] = {}
        self._worker_models: Dict[int, tuple] = {}
        self._job_retained_shm: Dict[int, List[str]] = {}
        self._leaf_shm_names: List[str] = []
        self._leaf_shm_owners: Dict[str, Any] = {}
        self._leaf_original_data: Dict[str, Any] = {}
        self._leaf_unshared_originals: List[tuple] = []
        self._task_results: Dict[str, ResultEntry] = {}

        self._owns_pool = worker_pool is None
        if worker_pool is None:
            threads_per_worker = default_threads_per_worker()
            worker_pool = PersistentWorkerPool(
                self.max_num_workers,
                _WORKER_DISPATCH,
                ctx=create_mp_context(),
                threads_per_worker=threads_per_worker,
                gpu_devices=list(getattr(self.scheduler, "gpu_devices", [])),
                gpu_slots_per_device=default_gpu_slots_per_device(),
                gpu_demand_fraction=1.0,
            )
        self._pool = worker_pool
        restrict_devices = getattr(self.scheduler, "restrict_gpu_devices", None)
        if callable(restrict_devices):
            restrict_devices(
                getattr(worker_pool, "gpu_worker_devices", []),
                getattr(worker_pool, "gpu_slots_per_device", 1),
            )

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

    def _should_materialize_leaf(self, modality: Any) -> bool:
        if not _materialize_lazy_leaves():
            return False
        data = modality.data if modality.has_data() else None
        if data is not None and not isinstance(data, LazyFileSequence):
            return False
        estimate = getattr(
            getattr(modality, "data_loader", None), "estimate_decoded_bytes", None
        )
        if not callable(estimate):
            return False
        metadata = None
        if data is not None and len(modality.metadata or []) == len(data):
            metadata = modality.metadata
        try:
            decoded_bytes = estimate(metadata)
        except (KeyError, TypeError, ValueError):
            return False
        if not decoded_bytes:
            return False
        budget = self.scheduler.memory_budget["cpu"] * _lazy_leaf_budget_fraction()
        if self._leaf_cached_bytes + decoded_bytes > budget:
            return False
        return decoded_bytes < SHARED_MEMORY_MIN_BYTES or _should_offload(decoded_bytes)

    def _load_leaf_modalities(self) -> None:
        for modality in self._modalities:
            if self._loads_in_chunks(modality):
                continue
            original = None
            if getattr(modality, "has_data", None) and modality.has_data():
                original = modality.data
                if not isinstance(original, LazyFileSequence):
                    wrapped, shm_name, _, _ = add_shared_memory_candidate(original)
                    if shm_name is not None:
                        modality._data = wrapped
                        self._leaf_shm_names.append(shm_name)
                        self._leaf_shm_owners[shm_name] = modality
                        self._leaf_original_data[shm_name] = original
                    continue
            materialize = self._should_materialize_leaf(modality)
            attempts = 0
            while True:
                self._pool.submit("load_leaf", (modality, materialize), gpu_id=None)
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
            if jr.value.get("materialized"):
                self._leaf_unshared_originals.append((modality, original))
            shm_name = jr.value.get("shm_name")
            if shm_name is not None:
                self._leaf_shm_names.append(shm_name)
                self._leaf_shm_owners[shm_name] = modality
                if original is not None:
                    self._leaf_original_data[shm_name] = original
                shm_bytes = int(jr.value.get("shm_bytes", 0) or 0)
                if shm_bytes:
                    # The decoded leaf now occupies RAM (/dev/shm) for the
                    # whole run; the scheduler only knew one lazy instance.
                    self._leaf_cached_bytes += shm_bytes
                    self._sync_cache_accounting()

    @staticmethod
    def _loads_in_chunks(modality: Any) -> bool:
        data_loader = getattr(modality, "data_loader", None)
        return getattr(data_loader, "chunk_size", None) is not None

    def _cleanup_leaf_shared_memory(self) -> None:
        for shm_name in self._leaf_shm_names:
            modality = self._leaf_shm_owners.pop(shm_name, None)
            if modality is not None and shm_name in collect_shm_names_from_payload(
                getattr(modality, "_data", None)
            ):
                close = getattr(modality._data, "close", None)
                if callable(close):
                    close()
                modality._data = self._leaf_original_data.pop(shm_name, None)
            unlink_shm(shm_name)
        self._leaf_shm_names = []
        self._leaf_shm_owners = {}
        for modality, original in self._leaf_unshared_originals:
            modality._data = original
        self._leaf_unshared_originals = []

    def _submit_node(
        self, node_id: str, allow_gpu_worker_for_cpu: bool = False
    ) -> None:
        node = self.scheduler.mapping[node_id]
        gpu_id = node.gpu_id
        parent_ids = self.scheduler.get_valid_parents(node_id)
        parent_results = (
            [self._get_cached_parent(pid) for pid in parent_ids] if parent_ids else None
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
                allow_gpu_worker_for_cpu=allow_gpu_worker_for_cpu,
            )
        else:
            payload = self._modalities if parent_results is None else parent_results
            retained = self._retain_for_submit(parent_ids, payload)
            self.scheduler.begin_execution(node_id)
            self.scheduler.move_to_running(node_id)
            cache_key = _operation_cache_key(node)
            job_id = self._pool.submit(
                "node",
                (node, payload),
                gpu_id=gpu_id,
                allow_gpu_worker_for_cpu=allow_gpu_worker_for_cpu,
                preferred_pids=[
                    pid
                    for pid, (_, _, key) in self._worker_models.items()
                    if cache_key is not None and key == cache_key
                ],
            )
            if cache_key is not None:
                self._set_worker_model(self._pool.job_pid(job_id), 0, None)

        self._job_units[job_id] = _NodeUnit(node_id)
        self._job_retained_shm[job_id] = retained

    def _submit_leaf_batch(
        self, node_ids: List[str], allow_gpu_worker_for_cpu: bool = False
    ) -> None:
        nodes = [self.scheduler.mapping[nid] for nid in node_ids]
        gpu_id = nodes[0].gpu_id
        retained = self._retain_for_submit([], self._modalities[0].data)
        for nid in node_ids:
            self.scheduler.begin_execution(nid)
        self.scheduler.move_to_running(node_ids)
        job_id = self._pool.submit(
            "leaf_batch",
            (nodes, self._modalities[0]),
            gpu_id=gpu_id,
            allow_gpu_worker_for_cpu=allow_gpu_worker_for_cpu,
        )
        self._job_units[job_id] = _BatchUnit(node_ids)
        self._job_retained_shm[job_id] = retained

    def _set_worker_model(
        self, pid: Optional[int], retained_bytes: int, cache_key: Optional[str]
    ) -> None:
        if pid is None:
            return
        if cache_key is None:
            self._worker_models.pop(pid, None)
        else:
            self._worker_models[pid] = (
                self._pool.worker_gpu(pid),
                int(retained_bytes),
                cache_key,
            )
        bytes_by_gpu: Dict[int, int] = {}
        for gpu_id, nbytes, _ in self._worker_models.values():
            if gpu_id is not None:
                bytes_by_gpu[gpu_id] = bytes_by_gpu.get(gpu_id, 0) + nbytes
        set_retained = getattr(self.scheduler, "set_retained_gpu_bytes", None)
        if callable(set_retained):
            set_retained(bytes_by_gpu)

    def _submit_model_release(self, pid: int) -> int:
        job_id = self._pool.submit_to(pid, "release_models", None)
        self._job_units[job_id] = _ReleaseUnit(pid)
        self._job_retained_shm[job_id] = []
        return job_id

    def _release_blocking_models(self) -> None:
        pop_requests = getattr(self.scheduler, "pop_release_requests", None)
        if not callable(pop_requests):
            return
        for gpu_id in pop_requests():
            for pid, (worker_gpu, nbytes, _) in list(self._worker_models.items()):
                if worker_gpu == gpu_id and nbytes > 0 and self._pool.is_idle(pid):
                    self._submit_model_release(pid)

    def _release_all_worker_models(self) -> None:
        """Leave a reused pool's GPUs as free as the next executor measures."""
        if self._job_units:
            return
        for pid in list(self._worker_models):
            if self._pool.is_idle(pid):
                self._submit_model_release(pid)
        while self._job_units:
            self._process_result(self._pool.wait())

    def _fill_pipeline(self) -> None:
        if not self._pool.has_idle_worker:
            return
        ready = self.scheduler.get_runnable().copy()

        def _gpu_id(entry):
            node_id = entry[0] if isinstance(entry, list) else entry
            return self.scheduler.mapping[node_id].gpu_id

        ready.sort(key=lambda entry: _gpu_id(entry) is None)
        for entry in ready:
            if not self._pool.has_idle_worker:
                break
            gpu_id = _gpu_id(entry)
            allow_gpu_worker_for_cpu = gpu_id is None
            if not self._pool.has_idle_worker_for(
                gpu_id,
                allow_gpu_worker_for_cpu=allow_gpu_worker_for_cpu,
            ):
                continue
            if isinstance(entry, list):
                self._submit_leaf_batch(
                    entry,
                    allow_gpu_worker_for_cpu=allow_gpu_worker_for_cpu,
                )
            else:
                if not self.scheduler.can_start_now(entry):
                    continue
                self._submit_node(
                    entry,
                    allow_gpu_worker_for_cpu=allow_gpu_worker_for_cpu,
                )
        self._release_blocking_models()

    def _get_cached_parent(self, parent_id: str):
        try:
            result = self._result_cache.get(parent_id)
        except KeyError:
            self._cache_misses += 1
            raise
        self._cache_consumptions += 1
        return result

    def _record_stats(self, node_id: str, pid: int, start_time: float, end_time: float):
        self._nodes_executed += 1
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
            if isinstance(unit, _ReleaseUnit):
                # A failed release is not retried; forgetting the model at
                # worst under-counts it until the worker's next node.
                retained_bytes = jr.value["retained_gpu_bytes"] if jr.ok else 0
                self._set_worker_model(unit.pid, retained_bytes, None)
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
            self._sync_cache_accounting()

    def _sync_cache_accounting(self) -> None:
        cache_bytes = self._result_cache.get_memory_total_memory_usage()
        self._peak_cache_bytes = max(self._peak_cache_bytes, cache_bytes)
        self.scheduler.set_cached_cpu_bytes(self._leaf_cached_bytes + cache_bytes)

    def _handle_job_failure(self, unit: Union[_NodeUnit, _BatchUnit], jr):
        if jr.worker_died or (
            isinstance(unit, _NodeUnit)
            and not self._is_task_node(self.scheduler.mapping[unit.node_id])
        ):
            self._set_worker_model(jr.pid, 0, None)
        reason = jr.error or "unknown worker failure"
        if jr.cuda_oom:
            reason += " (CUDA out of memory)"
        node_ids = unit.node_ids if isinstance(unit, _BatchUnit) else [unit.node_id]
        for node_id in node_ids:
            requeued = self._requeue_or_give_up(node_id, reason)
            if not requeued:
                self._release_parents(node_id)
            elif isinstance(unit, _NodeUnit) and jr.gpu_peak_bytes > 0:
                self.scheduler.raise_gpu_estimate(node_id, jr.gpu_peak_bytes)

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
            entry.train_fold_scores = value["scores"][0].fold_scores()
            entry.val_fold_scores = value["scores"][1].fold_scores()
            entry.test_fold_scores = value["scores"][2].fold_scores()
            entry.train_score_stds = value["scores"][0].score_stds()
            entry.val_score_stds = value["scores"][1].score_stds()
            entry.test_score_stds = value["scores"][2].score_stds()
            entry.task_timing = value.get("task_timing", {})
            entry.peak_cpu_bytes = value.get("peak_bytes", -1)
            entry.peak_gpu_bytes = value.get("gpu_peak_bytes", -1)
            entry.eval_index = self._eval_counter
            entry.t_since_search_start_s = time.perf_counter() - self._search_start
            entry.t_eval_end_unix = time.time()
            self._eval_counter += 1
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
            self._set_worker_model(
                value.get("pid"),
                value.get("retained_gpu_bytes", 0),
                value.get("cached_operation"),
            )
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
        shm_info = value.get("shm_info", {})
        for node_id, transformed_modality in results.items():
            info = shm_info.get(node_id, {})
            self._handle_modality_result(
                transformed_modality,
                node_id,
                None,
                None,
                self.scheduler.mapping[node_id].operation.__name__,
                actual_stats=info.get("actual_stats"),
                shm_name=info.get("shm_name"),
                resident_bytes=info.get("resident_bytes"),
                shm_bytes=info.get("shm_bytes", 0),
            )

        for node_id, reason in value.get("failed_nodes", {}).items():
            if not self._requeue_or_give_up(node_id, reason):
                self._release_parents(node_id)

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
                estimated_stats=estimated_stats,
            )
        self._manage_result_cache(
            node_id,
            transformed_modality,
            shm_name=shm_name,
            resident_bytes=resident_bytes,
            shm_bytes=shm_bytes,
        )
        self._sync_cache_accounting()
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
        estimated_stats: Optional[RepresentationStats] = None,
    ):
        if self.enable_checkpointing:
            self._memory_usage_checkpoint.increment(node_id)
        if estimated_stats is None:
            estimated_stats = self.scheduler.node_stats.get(node_id)

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
                "estimated_output_shape": getattr(
                    estimated_stats, "output_shape", None
                ),
                "estimated_num_instances": getattr(
                    estimated_stats, "num_instances", None
                ),
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

    def detach_worker_pool(self) -> PersistentWorkerPool:
        self._owns_pool = False
        return self._pool

    def run(self) -> Dict[str, Any]:
        self._task_results = {}
        try:
            self._load_leaf_modalities()
            while self._job_units or not self.scheduler.is_finished():
                self._fill_pipeline()
                if not self._job_units:
                    continue
                jr = self._pool.wait()
                self._process_result(jr)
        finally:
            if self._owns_pool:
                self._pool.shutdown()
            else:
                self._release_all_worker_models()
            self._result_cache.cleanup_all()
            self._cleanup_leaf_shared_memory()

        if self.enable_checkpointing:
            self._checkpoint_manager.save_checkpoint(
                self._task_results, self._discard_report()
            )

        self.statistics["reuse"] = self._reuse_report()

        return {
            "task_results": list(self._task_results.values()),
            "statistics": self.statistics,
        }

    def _reuse_report(self) -> Dict[str, Any]:
        report = dict(getattr(self.scheduler, "reuse_stats", {}) or {})
        slots = report.get("dag_node_slots", 0)
        saved = report.get("nodes_saved_by_merging", 0)
        report.update(
            {
                "nodes_executed": self._nodes_executed,
                "cache_consumptions": self._cache_consumptions,
                "cache_misses": self._cache_misses,
                "peak_cache_bytes": self._peak_cache_bytes,
                "slot_reduction_ratio": (saved / slots) if slots else 0.0,
                "failed_nodes": len(getattr(self.scheduler, "failed_nodes", []) or []),
                "blocked_memory_nodes": len(
                    getattr(self.scheduler, "blocked_memory_nodes_perm", []) or []
                ),
                "cpu_fallback_nodes": len(
                    getattr(self.scheduler, "cpu_fallback_nodes", []) or []
                ),
                "planned_task_nodes": len(getattr(self.scheduler, "roots", ()) or ()),
                "completed_task_nodes": self._eval_counter,
                "exclusive_memory_nodes": len(
                    getattr(self.scheduler, "exclusive_memory_nodes", {}) or {}
                ),
                "deadlock": bool(getattr(self.scheduler, "deadlock", False)),
                "deadlock_reason": getattr(self.scheduler, "deadlock_reason", None),
            }
        )
        return report
