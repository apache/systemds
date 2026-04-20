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
from systemds.scuro.utils.static_variables import DEBUG


class MemoryAwareNodeScheduler:

    def __init__(
        self,
        nodes: List[RepresentationDag],
        modalities: List[Modality],
        tasks: List[Any],
        cpu_memory_budget: float,
        gpu_memory_budget: Optional[float] = None,
    ):
        self.node_stats = {}
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
            "cpu_in_use": sum([self.node_resources[node][0] for node in self.leaves]),
            "gpu_in_use": {
                info["index"]: int(info["total_b"] - info["free_b"])
                for info in self.gpu_memory_info
            },
        }
        self._initialized = False

    def update_cpu_memory_in_use(self, delta_bytes: int):
        self.memory_stats["cpu_in_use"] += delta_bytes

    def get_runnable(self) -> List[RepresentationNode]:
        runnable_nodes = self._get_runnable_nodes()

        for node in runnable_nodes:
            ok, gpu_id = self._check_memory_constraints(node)
            if ok:
                self.mapping[node].gpu_id = gpu_id
                self.ready_nodes.append(node)
                self._reserve_memory(node, gpu_id)
        return self.ready_nodes

    def _get_runnable_nodes(self) -> List[str]:
        runnable_nodes = []
        for node in self.topo_order:
            if (
                node not in self.leaves
                and self.unresolved_parents[node] == 0
                and node not in self.running_nodes
                and node not in self.completed_nodes
                and node not in self.ready_nodes
            ):
                runnable_nodes.append(node)

        def _score(node_id: str):
            release_bytes = 0
            for parent_id in self.parents.get(node_id, set()):
                if (
                    parent_id not in self.leaves
                    and self.remaining_children.get(parent_id, 0) == 1
                ):
                    release_bytes += self.node_resources[parent_id][0]
            return (-release_bytes, node_id not in self.roots)

        runnable_nodes.sort(key=_score)
        return runnable_nodes

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

        for parent_id in self.parents.get(node_id, set()):
            if self.remaining_children.get(parent_id, 0) > 0:
                self.remaining_children[parent_id] -= 1

    def is_finished(self) -> bool:
        if not self._initialized:
            self._initialized = True
            return False

        if self.not_enough_memory():
            self.deadlock = True
            self.success = False
            return True

        if self._is_deadlock():
            self.deadlock = True
            self.success = False
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

    def update_node_stats_and_reestimate_descendants(
        self, node_id: str, actual_stats: Any
    ) -> None:
        if actual_stats is None:
            return

        self.node_stats[node_id] = actual_stats

        descendants = self._get_descendants_in_topo_order(node_id)

        for desc_id in descendants:
            if (
                desc_id in self.leaves
                or desc_id in self.ready_nodes
                or desc_id in self.running_nodes
                or desc_id in self.completed_nodes
            ):
                continue

            parent_ids = list(self.parents.get(desc_id, set()))
            if not parent_ids:
                continue

            input_stats = self.node_stats.get(parent_ids[0])
            if input_stats is None:
                continue

            if desc_id not in self.roots:
                operation = self.mapping[desc_id].operation(
                    params=self.mapping[desc_id].parameters
                )
                peak_memory = operation.estimate_peak_memory_bytes(input_stats)
                peak_memory["cpu_peak_bytes"] += (
                    64 * 1024 + 512 * input_stats.num_instances
                )
                output_stats = operation.get_output_stats(input_stats)

                self.node_resources[desc_id] = (
                    int(peak_memory["cpu_peak_bytes"]),
                    int(peak_memory["gpu_peak_bytes"]),
                )
                self.node_stats[desc_id] = output_stats
            else:
                task_node = self.mapping[desc_id]
                task = self.tasks[task_node.parameters["_task_idx"]]
                peak_memory = task.estimate_peak_memory_bytes(input_stats)

                cpu_increment = max(
                    int(peak_memory["cpu_peak_bytes"]),
                    16 * 1024 * 1024,
                )
                self.node_resources[desc_id] = (
                    cpu_increment,
                    int(peak_memory["gpu_peak_bytes"]),
                )
                self.node_stats[desc_id] = task.get_output_stats(input_stats)

    def _get_descendants_in_topo_order(self, node_id: str) -> List[str]:
        reachable = set()
        queue = deque(self.children.get(node_id, set()))
        while queue:
            cur = queue.popleft()
            if cur in reachable:
                continue
            reachable.add(cur)
            for child in self.children.get(cur, set()):
                queue.append(child)

        return [nid for nid in self.topo_order if nid in reachable]

    def _is_success(self) -> bool:
        return (
            len(self.running_nodes) == 0
            and self.ready_nodes == []
            and len(self._get_pending_nodes()) == 0
        )

    def _is_deadlock(self) -> bool:
        pending_nodes = self._get_pending_nodes()
        blocked = len(pending_nodes) > 0
        # if len(self.running_nodes) == 0 and len(self.ready_nodes) == 0 and blocked:
        #     return True

        for node_id in pending_nodes:
            if node_id not in self.blocked_memory_nodes_perm:
                blocked = False
                break

        return blocked

    def not_enough_memory(self) -> bool:
        for node_id in self._get_pending_nodes():
            cpu_mem, gpu_mem = self.node_resources[node_id]
            if cpu_mem > self.memory_budget["cpu"] - self.memory_stats["cpu_in_use"]:
                return True
            if gpu_mem > 0.0 and self.n_gpu > 0:
                gpu_id = self._gpu_with_most_free_memory(gpu_mem)
                if gpu_id is None:
                    return True
        return self.memory_stats["cpu_in_use"] > self.memory_budget["cpu"]

    def _check_memory_constraints(self, node_id: str) -> bool:
        cpu_mem, gpu_mem = self.node_resources[node_id]
        gpu_id = None
        if cpu_mem > self.memory_budget["cpu"] - self.memory_stats["cpu_in_use"]:
            if cpu_mem > self.memory_budget["cpu"]:
                self.blocked_memory_nodes_perm.append(node_id)
                self.topo_order.remove(node_id)
            return False, None

        if gpu_mem > 0.0 and self.n_gpu > 0:
            gpu_id = self._gpu_with_most_free_memory(gpu_mem)

            if gpu_id is None:
                if DEBUG:
                    print(f"Node {node_id} has no available GPU")
                return False, None

        return True, gpu_id

    def _gpu_with_most_free_memory(self, memory_needed):
        free_memory = []
        for i in range(self.n_gpu):
            free_memory.append(
                self.memory_budget["gpu"][i] - self.memory_stats["gpu_in_use"][i]
            )

        if max(free_memory) < memory_needed:
            return None

        return free_memory.index(max(free_memory))

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

        self.remaining_children = {
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
                    peak_memory["cpu_peak_bytes"] += (
                        64 * 1024 + 512 * input_stats.num_instances
                    )  # Placeholder for transformed modality creation overhead
                    peak_memory["cpu_peak_bytes"] *= 1
                    output_stats = operation.get_output_stats(input_stats)
                    node_resources[node] = (
                        int(peak_memory["cpu_peak_bytes"]),
                        int(peak_memory["gpu_peak_bytes"]),
                    )
                    node_stats[node] = output_stats
                else:
                    task_node = self.mapping[node]
                    task = self.tasks[task_node.parameters["_task_idx"]]
                    peak_memory = task.estimate_peak_memory_bytes(input_stats)

                    cpu_increment = max(
                        int(peak_memory["cpu_peak_bytes"]),
                        16 * 1024 * 1024,
                    )
                    node_resources[node] = (
                        cpu_increment,
                        int(peak_memory["gpu_peak_bytes"]),
                    )

                    node_stats[node] = task.get_output_stats(input_stats)
        self.node_stats = node_stats
        return node_resources

    @staticmethod
    def _stats_to_bytes(stats: Optional[Any], dtype_size: int = 4) -> int:
        if stats is None:
            return 0
        num_instances = int(getattr(stats, "num_instances", 0))
        output_shape = tuple(getattr(stats, "output_shape", ()))
        numel = 1
        for dim in output_shape:
            try:
                numel *= int(dim)
            except Exception:
                numel *= 1
        return max(0, int(num_instances * numel * dtype_size))
