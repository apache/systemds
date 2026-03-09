from __future__ import annotations

from typing import List, Dict, Optional, Any
from collections import defaultdict, deque
from collections import deque
import torch
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationNode,
)
from systemds.scuro.modality.modality import Modality
from systemds.scuro.utils.memory_utility import gpu_memory_info


class MemoryAwareNodeScheduler:

    def __init__(
        self,
        nodes: List[RepresentationDag],
        modalities: List[Modality],
        tasks: List[Any],
        cpu_memory_budget: float,
        gpu_memory_budget: Optional[float] = None,
    ):
        self.modalities = modalities
        self.tasks = tasks
        self.parents = defaultdict(set)
        self.children = defaultdict(set)
        self.mapping = {}
        self.parent_refcounts = defaultdict(int)
        self.gpu_memory_info = gpu_memory_info()
        self.memory_budget = {
            "cpu": cpu_memory_budget,
            "gpu": {
                info["index"]: int(info["total_b"]) for info in self.gpu_memory_info
            },
        }
        self.roots = set()
        self.leaves = set()
        self.topo_order = []
        self.nodes = self._get_nodes_from_dags(nodes)
        self.unresolved_parents = self._get_unresolved_parents()
        self.node_resources = self._estimate_node_resources()
        self.success = False
        self.deadlock = False
        self.ready_nodes = []
        self.running_nodes = []
        self.completed_nodes = []
        self.failed_nodes = []
        self.blocked_memory_nodes_perm = []
        self.cancelled_nodes = []
        self.n_gpu = (
            torch.cuda.device_count() if torch and torch.cuda.is_available() else 0
        )
        self.memory_stats = {
            "cpu_in_use": 0.0,
            "gpu_in_use": {
                info["index"]: int(info["total_b"] - info["free_b"])
                for info in self.gpu_memory_info
            },
        }

    def get_runnable(self) -> List[RepresentationNode]:
        # TODO: prioritize task nodes over representation nodes to free up memory in cache if possible
        for node in self.topo_order:
            if node not in self.leaves and self.unresolved_parents[node] == 0:
                ok, gpu_id = self._check_memory_constraints(node)
                if ok:
                    self.mapping[node].gpu_id = gpu_id
                    self.ready_nodes.append(node)
                    self._reserve_memory(node, gpu_id)
        return self.ready_nodes

    def add_failed_node(self, node_id: str):
        self.failed_nodes.append(node_id)
        self.running_nodes.remove(node_id)

        self._release_memory(node_id, self.mapping[node_id].gpu_id)

    def move_to_running(self, node_id: str):
        self.ready_nodes.remove(node_id)
        self.running_nodes.append(node_id)

    def complete_node(self, node_id: str):
        self.running_nodes.remove(node_id)
        self.completed_nodes.append(node_id)
        self._release_memory(node_id, self.mapping[node_id].gpu_id)
        self.topo_order.remove(node_id)
        for child_id in self.children[node_id]:
            self.parent_refcounts[child_id] -= 1
            self.unresolved_parents[child_id] -= 1

    def is_finished(self) -> bool:
        if self._is_deadlock():
            self.deadlock = True
            return True

        if self._is_success():
            self.success = True
            return True

        return False

    def get_valid_parent(self, node_id: str) -> bool:
        for parent_id in self.parents[node_id]:
            if parent_id not in self.leaves:
                return parent_id
        return None

    def get_children(self, node_id: str) -> List[str]:
        return list(self.children[node_id])

    def _is_success(self) -> bool:
        return (
            len(self.running_nodes) == 0
            and self.ready_nodes == []
            and len(self._get_pending_nodes()) == 0
        )

    def _is_deadlock(self) -> bool:
        pending_nodes = self._get_pending_nodes()
        blocked = len(pending_nodes) > 0

        for node_id in pending_nodes:
            if node_id not in self.blocked_memory_nodes_perm:
                blocked = False
                break

        return len(self.running_nodes) == 0 and self.ready_nodes == [] and blocked

    def _check_memory_constraints(self, node_id: str) -> bool:
        cpu_mem, gpu_mem = self.node_resources[node_id]
        gpu_id = None
        if cpu_mem > self.memory_budget["cpu"] - self.memory_stats["cpu_in_use"]:
            if cpu_mem > self.memory_budget["cpu"]:
                self.blocked_memory_nodes_perm.append(node_id)
            return False, None

        if gpu_mem > 0.0:
            for i in range(self.n_gpu):
                if (
                    gpu_mem
                    < self.memory_budget["gpu"][i] - self.memory_stats["gpu_in_use"][i]
                ):
                    gpu_id = i
                    break
            if gpu_id is None:
                print(f"Node {node_id} has no available GPU")
                return False, None

        return True, gpu_id

    def _get_pending_nodes(self) -> List[str]:
        return [
            node_id
            for node_id in self.topo_order
            if node_id not in self.leaves and self.unresolved_parents[node_id] == 0
        ]

    def _reserve_memory(self, node_id: str, gpu_id: int) -> bool:
        cpu_mem, gpu_mem = self.node_resources[node_id]

        self.memory_stats["cpu_in_use"] += cpu_mem
        if gpu_id is not None:
            self.memory_stats["gpu_in_use"][gpu_id] += gpu_mem

    def _release_memory(self, node_id: str, gpu_id: int) -> bool:
        cpu_mem, gpu_mem = self.node_resources[node_id]
        self.memory_stats["cpu_in_use"] -= cpu_mem
        if gpu_id is not None:
            self.memory_stats["gpu_in_use"][gpu_id] -= gpu_mem

    def _get_nodes_from_dags(
        self, dags: List[RepresentationDag]
    ) -> List[RepresentationNode]:
        nodes = []
        for dag in dags:
            self.roots.add(dag.root_node_id)
            for node in dag.nodes:
                self.mapping[node.node_id] = node
                if not node.inputs:
                    self.leaves.add(node.node_id)
                for parent_id in node.inputs:
                    self.parents[node.node_id].add(parent_id)
                    self.children[parent_id].add(node.node_id)

        indegree = {
            node_id: len(self.parents.get(node_id, set())) for node_id in self.mapping
        }
        queue = deque([node_id for node_id, deg in indegree.items() if deg == 0])
        topo_order: List[str] = []
        while queue:
            node_id = queue.popleft()
            topo_order.append(node_id)
            for child_id in self.children.get(node_id, set()):
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    queue.append(child_id)
        if len(topo_order) != len(self.mapping):
            raise ValueError("Node graph contains a cycle")
        self.topo_order = topo_order
        self.parent_refcounts = {
            node_id: len(self.children.get(node_id, set())) for node_id in self.mapping
        }
        return nodes

    def _get_unresolved_parents(self) -> Dict[str, int]:
        unresolved_parents = {}
        for node_id in self.mapping:
            unresolved_parents[node_id] = sum(
                1
                for parent_id in self.parents.get(node_id, set())
                if parent_id not in self.leaves
            )
        return unresolved_parents

    def _get_modality_from_id(self, modality_id: str) -> Modality:
        for modality in self.modalities:
            if modality.modality_id == modality_id:
                return modality
        return None

    def _estimate_node_resources(self):
        node_resources = {}
        node_stats = {}
        for node in self.topo_order:
            if node in self.leaves:
                modality = self._get_modality_from_id(self.mapping[node].modality_id)
                peak_memory = modality.estimate_peak_memory_bytes()
                node_resources[node] = (
                    peak_memory["cpu_peak_bytes"],
                    peak_memory["gpu_peak_bytes"],
                )
                node_stats[node] = modality.get_stats()
            else:
                parent_ids = list(self.parents.get(node, set()))
                parent_stats = [node_stats[parent_id] for parent_id in parent_ids]
                input_stats = parent_stats[0] if parent_stats else None
                if node not in self.roots:
                    operation = self.mapping[node].operation(
                        params=self.mapping[node].parameters
                    )
                    peak_memory = operation.estimate_peak_memory_bytes(input_stats)
                    node_resources[node] = (
                        peak_memory["cpu_peak_bytes"],
                        peak_memory["gpu_peak_bytes"],
                    )
                    node_stats[node] = operation.get_output_stats(input_stats)
                else:
                    task_node = self.mapping[node]
                    task = self.tasks[task_node.parameters["_task_idx"]]
                    peak_memory = task.estimate_peak_memory_bytes(input_stats)
                    node_resources[node] = (
                        peak_memory["cpu_peak_bytes"],
                        peak_memory["gpu_peak_bytes"],
                    )

                    node_stats[node] = task.get_output_stats(input_stats)
        return node_resources
