from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
import os
from systemds.scuro import Modality
from systemds.scuro.drsearch.node_scheduler import MemoryAwareNodeScheduler
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationNode,
)
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
from pympler import asizeof


class RefCountResultCache:
    def __init__(self):
        self.cache = {}
        self.ref_count = {}

    def get(self, node_id: str) -> Any:
        return self.cache[node_id]

    def add_result(self, node_id: str, result: Any):
        self.cache[node_id] = result

    def inc_ref(self, node_id: str):
        if node_id not in self.ref_count:
            self.ref_count[node_id] = 0
        self.ref_count[node_id] += 1

    def dec_ref(self, node_id: str):
        self.ref_count[node_id] -= 1
        if self.ref_count[node_id] == 0:
            del self.cache[node_id]
            del self.ref_count[node_id]

    def clear(self):
        self.cache.clear()
        self.ref_count.clear()

    def __len__(self):
        return len(self.cache)


def _execute_node_worker(
    node: RepresentationNode,
    input_mods: List[Any],
    task: Any,
    rep_cache: Optional[Dict[str, Any]],
    gpu_id: Optional[int],
):
    print(
        f"Executing node {node.node_id} inputs: {input_mods[0].modality_id}, gpu: {gpu_id}"
    )
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
    print(f"Executing task {task.model.name} on GPU {gpu_id}")
    if gpu_id is not None and hasattr(task, "model") and hasattr(task.model, "device"):
        task.model.device = torch.device(f"cuda:{gpu_id}")
    start = time.perf_counter()
    scores = task.run(data)
    end = time.perf_counter()
    return {"scores": scores, "task_time": end - start}


# TODO: add a checkpoint manager only to the node executor, maybe get the name from outside to distinguish between unimodal and multimodal checkpoint managers
# we can exclude all dag nodes that are loaded through an existing checkpoint and therefore speedup the further execution
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

    def run(self) -> None:
        task_results = {}
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

                    before_bytes = self._result_cache_size_bytes()
                    self._manage_result_cache(node_id, result)
                    after_bytes = self._result_cache_size_bytes()
                    self.scheduler.update_cpu_memory_in_use(after_bytes - before_bytes)
                    self.scheduler.complete_node(node_id)

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
                    submit_new_ready_nodes()

        return list(task_results.values())

    def _result_cache_size_bytes(self) -> int:
        return asizeof.asizeof(self.result_cache.cache) + asizeof.asizeof(
            self.result_cache.ref_count
        )

    def _manage_result_cache(self, node_id: str, result: Any):
        parent_node_id = self.scheduler.get_valid_parent(node_id)
        if parent_node_id is not None:
            self.result_cache.dec_ref(parent_node_id)

        if self.scheduler.get_children(node_id):
            for _ in self.scheduler.get_children(node_id):
                self.result_cache.inc_ref(node_id)
            self.result_cache.add_result(node_id, result)

    def _execute_batch(self, ready_nodes_ids: List[str]) -> None:
        task_results = {}

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=self.max_num_workers, mp_context=ctx
        ) as executor:
            future_to_node_id = {}
            for node_id in ready_nodes_ids:
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

            for future in as_completed(future_to_node_id):
                node_id = future_to_node_id[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error executing node {node_id}: {e}")
                    self.scheduler.add_failed_node(node_id)
                    continue

                self.scheduler.complete_node(node_id)
                self._manage_result_cache(node_id, result)
                node = self.scheduler.mapping[node_id]
                if self._is_task_node(node):
                    task_results[node_id].task_time = result["task_time"]
                    task_results[node_id].train_score = result["scores"][
                        0
                    ].average_scores
                    task_results[node_id].val_score = result["scores"][1].average_scores
                    task_results[node_id].test_score = result["scores"][
                        2
                    ].average_scores

        return task_results

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
