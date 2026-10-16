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
import os
import re
from typing import List, Dict, Optional, Any
from collections import defaultdict, deque
from collections import deque
import torch
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationNode,
)
from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.representation import (
    stats_dtype,
    stats_itemsize,
)
from systemds.scuro.utils.memory_utility import gpu_memory_info
from systemds.scuro.utils.static_variables import DEBUG

_MAX_GPU_SCHEDULE_ATTEMPTS = int(os.environ.get("SCURO_MAX_GPU_SCHEDULE_ATTEMPTS", "3"))


def _exclusive_over_budget_enabled() -> bool:
    return os.environ.get("SCURO_EXCLUSIVE_OVER_BUDGET", "0") == "1"


def _cache_drain_fraction() -> float:
    return float(os.environ.get("SCURO_CACHE_DRAIN_FRACTION", "0.5"))


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
        self.node_costs = self._estimate_node_costs()
        self.node_priorities = self._compute_upward_ranks()
        self.gpu_devices = list(self.memory_budget["gpu"])
        self.gpu_slots_per_device = max(
            1, int(os.environ.get("SCURO_GPU_SLOTS_PER_DEVICE", "1"))
        )
        self.gpu_slots_in_use = {gpu_id: 0 for gpu_id in self.gpu_devices}
        self._gpu_slot_nodes: Dict[str, int] = {}
        self.success = False
        self.deadlock = False
        self.ready_nodes = []
        self._ready_set = set()
        self.running_nodes = set()
        self.completed_nodes = []
        self._completed_set = set()
        self.failed_nodes = []
        self.failed_node_reasons: Dict[str, str] = {}
        self.blocked_memory_nodes_perm = []
        self.blocked_memory_reasons: Dict[str, str] = {}
        self.cancelled_nodes = []
        self.deadlock_reason: Optional[str] = None
        self.gpu_wait_attempts: Dict[str, int] = {}
        self.cpu_fallback_nodes: List[str] = []
        self.cpu_fallback_reasons: Dict[str, str] = {}
        self._candidates = {
            node_id
            for node_id in self.topo_order
            if node_id not in self.leaves and self.unresolved_parents[node_id] == 0
        }
        self.n_gpu = len(self.gpu_devices)
        leaf_cached = sum(self.node_resources[node][0] for node in self.leaves)
        self.memory_stats = {
            "cpu_cached": leaf_cached,
            "cpu_in_flight": 0,
            "gpu_in_use": {
                info["index"]: int(info["total_b"] - info["free_b"])
                for info in self.gpu_memory_info
            },
        }
        self._cpu_reserved_nodes: Dict[str, int] = {}
        self.exclusive_memory_nodes: Dict[str, str] = {}
        self._initialized = False

    def _total_cpu_in_use(self) -> float:
        return self.memory_stats["cpu_cached"] + self.memory_stats["cpu_in_flight"]

    def _pending_admitted_cpu_bytes(self) -> int:
        total = 0
        for node_id in self._ready_set:
            if node_id in self._cpu_reserved_nodes:
                continue
            resources = self.node_resources.get(node_id)
            if resources:
                total += resources[0]
        return int(total)

    def can_start_now(self, node_id: str) -> bool:
        resources = self.node_resources.get(node_id)
        if not resources:
            return True
        if not self._cpu_reserved_nodes:
            return True
        return resources[0] <= self.memory_budget["cpu"] - self._total_cpu_in_use()

    def update_cpu_memory_in_use(self, delta_bytes: int):
        self.memory_stats["cpu_cached"] += delta_bytes

    def set_cached_cpu_bytes(self, cached_bytes: int) -> None:
        self.memory_stats["cpu_cached"] = cached_bytes

    def _cache_draining(self) -> bool:
        budget = self.memory_budget["cpu"]
        if _exclusive_over_budget_enabled() and self._total_cpu_in_use() >= budget:
            return True
        return self.memory_stats["cpu_cached"] >= _cache_drain_fraction() * budget

    def _release_bytes(self, node_id: str) -> int:
        release_bytes = 0
        for parent_id in self.parents.get(node_id, set()):
            if (
                parent_id not in self.leaves
                and self.remaining_children.get(parent_id, 0) == 1
            ):
                release_bytes += self.node_resources[parent_id][0]
        return release_bytes

    def _grows_cache(self, node_id: str) -> bool:
        adds_bytes = (
            self.node_resources[node_id][0] if self.children.get(node_id) else 0
        )
        return adds_bytes > self._release_bytes(node_id)

    def get_runnable(self) -> List[RepresentationNode]:
        runnable_nodes = self._get_runnable_nodes()

        admitted_bytes = self._pending_admitted_cpu_bytes()
        pending_gpu = {gpu_id: 0 for gpu_id in self.gpu_devices}
        pending_slots = {gpu_id: 0 for gpu_id in self.gpu_devices}
        draining = self._cache_draining()
        hold_growing_nodes = False

        for node in runnable_nodes:
            if node in self._ready_set:
                continue
            if hold_growing_nodes and self._grows_cache(node):
                continue
            ok, gpu_id = self._check_memory_constraints(
                node, admitted_bytes, pending_gpu, pending_slots
            )
            if ok:
                admitted_bytes += self.node_resources[node][0]
                if gpu_id is not None:
                    pending_gpu[gpu_id] += self.node_resources[node][1]
                    pending_slots[gpu_id] += 1
                self.mapping[node].gpu_id = gpu_id
                self._candidates.discard(node)
                self.ready_nodes.append(node)
                self._ready_set.add(node)
            elif (
                draining
                and not self._grows_cache(node)
                and self.node_resources[node][0]
                > self.memory_budget["cpu"] - self._total_cpu_in_use() - admitted_bytes
            ):
                hold_growing_nodes = True
        chunked_leaf_by_device = defaultdict(list)
        for node in self.ready_nodes:
            if isinstance(node, list):
                continue
            if any(re.fullmatch(r"leaf_\d+", i) for i in self.mapping[node].inputs):
                for mod in self.modalities:
                    if (
                        mod.modality_id
                        == self.mapping[self.mapping[node].inputs[0]].modality_id
                    ):
                        if mod.data_loader.chunk_size is not None:
                            chunked_leaf_by_device[self.mapping[node].gpu_id].append(
                                node
                            )
                            break

        for nodes_for_device in chunked_leaf_by_device.values():
            for node in nodes_for_device:
                self.ready_nodes.remove(node)

        self.ready_nodes.extend(chunked_leaf_by_device.values())
        return self.ready_nodes

    def _get_runnable_nodes(self) -> List[str]:
        runnable_nodes = list(self._candidates)

        # Upward rank always puts a producer ahead of its consumers, so without
        # a counterweight every large intermediate is materialized before the
        # tasks that would release it. Once the cache holds a large share of
        # the budget (or, with exclusive admission, the budget is exceeded),
        # nodes whose completion does not grow the cache (task roots, or last
        # children releasing at least their own output) run first.
        draining_first = self._cache_draining()

        def _score(node_id: str):
            release_bytes = self._release_bytes(node_id)
            grows_cache = draining_first and self._grows_cache(node_id)
            return (
                grows_cache,
                -self.node_priorities.get(node_id, 0.0),
                -release_bytes,
                node_id not in self.roots,
                node_id,
            )

        runnable_nodes.sort(key=_score)
        return runnable_nodes

    def _estimate_node_costs(self) -> Dict[str, float]:
        costs: Dict[str, float] = {}
        for node_id in self.topo_order:
            if node_id in self.leaves:
                costs[node_id] = 1.0
                continue
            if node_id in self.roots:
                parent_ids = list(self.parents.get(node_id, set()))
                parent_stats = [
                    self.node_stats.get(parent_id) for parent_id in parent_ids
                ]
                input_stats = (
                    parent_stats[0] if len(parent_stats) == 1 else parent_stats
                )
                task_idx = self.mapping[node_id].parameters.get("_task_idx", 0)
                try:
                    costs[node_id] = max(
                        1.0,
                        float(self.tasks[task_idx].estimate_relative_cost(input_stats)),
                    )
                except Exception:
                    costs[node_id] = 1.0
            else:
                cpu_bytes, gpu_bytes = self.node_resources.get(node_id, (0, 0))
                costs[node_id] = max(1.0, float(cpu_bytes + gpu_bytes) / (1024**2))
        return costs

    def _compute_upward_ranks(self) -> Dict[str, float]:
        ranks: Dict[str, float] = {}
        for node_id in reversed(self.topo_order):
            downstream = [ranks[child] for child in self.children.get(node_id, set())]
            ranks[node_id] = float(self.node_costs.get(node_id, 1.0)) + (
                max(downstream) if downstream else 0.0
            )
        return ranks

    def gpu_demand_fraction(self) -> float:
        total = 0.0
        gpu_total = 0.0
        for node_id, resources in self.node_resources.items():
            weight = float(self.node_costs.get(node_id, 1.0))
            if weight <= 0:
                weight = 1.0
            total += weight
            if resources[1] > 0:
                gpu_total += weight
        return gpu_total / total if total else 0.0

    def add_failed_node(self, node_id: str, reason: str = "unknown failure"):
        self.failed_nodes.append(node_id)
        self.failed_node_reasons[node_id] = reason
        self.running_nodes.discard(node_id)
        self._release_execution_memory(node_id, self.mapping[node_id].gpu_id)

    def requeue_node(self, node_id: str) -> None:
        self.running_nodes.discard(node_id)
        self._release_execution_memory(node_id, self.mapping[node_id].gpu_id)
        self._candidates.add(node_id)

    def begin_execution(self, node_id: str) -> None:
        gpu_id = self.mapping[node_id].gpu_id
        cpu_mem, gpu_mem = self.node_resources[node_id]
        if gpu_id is not None and gpu_mem > 0 and node_id not in self._gpu_slot_nodes:
            self.memory_stats["gpu_in_use"][gpu_id] += gpu_mem
            self.gpu_slots_in_use[gpu_id] += 1
            self._gpu_slot_nodes[node_id] = gpu_id
        if cpu_mem > 0 and node_id not in self._cpu_reserved_nodes:
            self._cpu_reserved_nodes[node_id] = int(cpu_mem)
            self.memory_stats["cpu_in_flight"] += int(cpu_mem)

    def move_to_running(self, node_id: str | list):
        self.ready_nodes.remove(node_id)
        if isinstance(node_id, list):
            self._ready_set.difference_update(node_id)
            self.running_nodes.update(node_id)
        else:
            self._ready_set.discard(node_id)
            self.running_nodes.add(node_id)

    def complete_node(self, node_id: str):
        self.running_nodes.discard(node_id)
        self.completed_nodes.append(node_id)
        self._completed_set.add(node_id)
        self._release_execution_memory(node_id, self.mapping[node_id].gpu_id)
        for child_id in self.children[node_id]:
            self.parent_refcounts[child_id] -= 1
            self.unresolved_parents[child_id] -= 1
            if (
                self.unresolved_parents[child_id] == 0
                and child_id not in self.leaves
                and child_id not in self._completed_set
            ):
                self._candidates.add(child_id)

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

    def get_valid_parents(self, node_id: str) -> bool:
        parents = []
        for parent_id in self.parents[node_id]:
            if parent_id not in self.leaves:
                parents.append(parent_id)
        return parents

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
                or desc_id in self._ready_set
                or desc_id in self.running_nodes
                or desc_id in self._completed_set
            ):
                continue

            parent_ids = list(self.parents.get(desc_id, set()))
            if not parent_ids:
                continue

            parent_stats = [self.node_stats.get(pid) for pid in parent_ids]
            if any(s is None for s in parent_stats):
                continue
            input_stats = parent_stats[0] if len(parent_stats) == 1 else parent_stats

            if desc_id not in self.roots:
                operation = self.mapping[desc_id].operation(
                    params=self.mapping[desc_id].parameters
                )
                peak_memory = operation.estimate_peak_memory_bytes(input_stats)
                input_stats_for_overhead = self._stats_for_overhead(input_stats)
                peak_memory["cpu_peak_bytes"] += (
                    64 * 1024 + 512 * input_stats_for_overhead.num_instances
                )
                output_stats = self._resolve_output_stats(
                    operation.get_output_stats(input_stats), input_stats
                )

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
        if self.running_nodes or self.ready_nodes:
            return False
        for node_id in self._candidates:
            cpu_mem, gpu_mem = self.node_resources[node_id]
            if (
                cpu_mem > self.memory_budget["cpu"] - self._total_cpu_in_use()
                and not _exclusive_over_budget_enabled()
            ):
                self.deadlock_reason = (
                    f"node {node_id} needs {cpu_mem / 1024**3:.2f} GB CPU but only "
                    f"{(self.memory_budget['cpu'] - self._total_cpu_in_use()) / 1024**3:.2f} "
                    f"GB is free and nothing is running to release more"
                )
                return True
            if gpu_mem > 0.0 and self.n_gpu > 0:
                gpu_id = self._gpu_with_most_free_memory(gpu_mem)
                if gpu_id is None:
                    if (
                        self.gpu_wait_attempts.get(node_id, 0)
                        <= _MAX_GPU_SCHEDULE_ATTEMPTS
                    ):
                        continue
                    self.deadlock_reason = (
                        f"node {node_id} needs {gpu_mem / 1024**3:.2f} GB GPU memory "
                        f"but no GPU has enough free and nothing is running to release more"
                    )
                    return True
        return False

    def _check_memory_constraints(
        self,
        node_id: str,
        pending_bytes: int = 0,
        pending_gpu: Optional[Dict[int, int]] = None,
        pending_slots: Optional[Dict[int, int]] = None,
    ) -> bool:
        cpu_mem, gpu_mem = self.node_resources[node_id]
        gpu_id = None
        free_cpu = self.memory_budget["cpu"] - self._total_cpu_in_use() - pending_bytes
        if (
            cpu_mem > free_cpu
            and _exclusive_over_budget_enabled()
            and (
                not self.running_nodes
                and not self._ready_set
                and pending_bytes == 0
                and self.memory_stats["cpu_in_flight"] == 0
            )
        ):
            self.exclusive_memory_nodes[node_id] = (
                f"estimated {cpu_mem / 1024**3:.2f} GB with {free_cpu / 1024**3:.2f} GB "
                f"of the CPU budget free; run alone"
            )
        elif cpu_mem > free_cpu:
            if (
                cpu_mem > self.memory_budget["cpu"]
                and not _exclusive_over_budget_enabled()
            ):
                self.blocked_memory_nodes_perm.append(node_id)
                self.blocked_memory_reasons[node_id] = (
                    f"estimated CPU peak {cpu_mem / 1024**3:.2f} GB exceeds the "
                    f"total CPU memory budget of "
                    f"{self.memory_budget['cpu'] / 1024**3:.2f} GB"
                )
                self._candidates.discard(node_id)
            return False, None

        if gpu_mem > 0.0 and self.n_gpu > 0:
            gpu_id = self._gpu_with_most_free_memory(
                gpu_mem, pending_gpu, pending_slots
            )

            if gpu_id is None:
                if self._waiting_for_gpu_slot(gpu_mem, pending_gpu, pending_slots):
                    return False, None
                attempts = self.gpu_wait_attempts.get(node_id, 0) + 1
                self.gpu_wait_attempts[node_id] = attempts
                if attempts > _MAX_GPU_SCHEDULE_ATTEMPTS:
                    reason = (
                        f"no GPU had {gpu_mem / 1024**3:.2f} GB free after "
                        f"{attempts} scheduling attempts; running on CPU instead"
                    )
                    if DEBUG:
                        print(f"Node {node_id}: {reason}")
                    self.cpu_fallback_nodes.append(node_id)
                    self.cpu_fallback_reasons[node_id] = reason
                    return True, None
                if DEBUG:
                    print(f"Node {node_id} has no available GPU")
                return False, None

        return True, gpu_id

    def _gpu_with_most_free_memory(
        self,
        memory_needed: int,
        pending_gpu: Optional[Dict[int, int]] = None,
        pending_slots: Optional[Dict[int, int]] = None,
    ) -> Optional[int]:
        pending_gpu = pending_gpu or {}
        pending_slots = pending_slots or {}
        candidates = []
        for gpu_id in self.gpu_devices:
            used_slots = self.gpu_slots_in_use.get(gpu_id, 0)
            used_slots += pending_slots.get(gpu_id, 0)
            if used_slots >= self.gpu_slots_per_device:
                continue
            free = self.memory_budget["gpu"][gpu_id]
            free -= self.memory_stats["gpu_in_use"].get(gpu_id, 0)
            free -= pending_gpu.get(gpu_id, 0)
            if free >= memory_needed:
                candidates.append((free, -used_slots, -gpu_id, gpu_id))
        if not candidates:
            return None
        return max(candidates)[-1]

    def _waiting_for_gpu_slot(
        self,
        memory_needed: int,
        pending_gpu: Optional[Dict[int, int]] = None,
        pending_slots: Optional[Dict[int, int]] = None,
    ) -> bool:
        pending_gpu = pending_gpu or {}
        pending_slots = pending_slots or {}
        for gpu_id in self.gpu_devices:
            free = self.memory_budget["gpu"][gpu_id]
            free -= self.memory_stats["gpu_in_use"].get(gpu_id, 0)
            free -= pending_gpu.get(gpu_id, 0)
            if free >= memory_needed:
                return True
        return False

    def restrict_gpu_devices(
        self, devices: List[int], slots_per_device: int = 1
    ) -> None:
        self.gpu_devices = [
            gpu_id for gpu_id in devices if gpu_id in self.memory_budget["gpu"]
        ]
        self.n_gpu = len(self.gpu_devices)
        self.gpu_slots_per_device = max(1, int(slots_per_device))
        self.gpu_slots_in_use = {gpu_id: 0 for gpu_id in self.gpu_devices}

    def _get_pending_nodes(self) -> List[str]:
        return list(self._candidates)

    def _release_execution_memory(self, node_id: str, gpu_id: int) -> None:
        _, gpu_mem = self.node_resources[node_id]
        reserved_gpu_id = self._gpu_slot_nodes.pop(node_id, None)
        if reserved_gpu_id is not None and gpu_mem > 0:
            self.memory_stats["gpu_in_use"][reserved_gpu_id] = max(
                0, self.memory_stats["gpu_in_use"][reserved_gpu_id] - gpu_mem
            )
            self.gpu_slots_in_use[reserved_gpu_id] = max(
                0, self.gpu_slots_in_use[reserved_gpu_id] - 1
            )
        reserved = self._cpu_reserved_nodes.pop(node_id, 0)
        if reserved:
            self.memory_stats["cpu_in_flight"] = max(
                0, self.memory_stats["cpu_in_flight"] - reserved
            )

    @staticmethod
    def _operation_label(node) -> str:
        """A readable name for what a node computes.

        `operation` is a class on representation nodes but an instance on some
        others, and task roots carry their kind in `parameters` instead, so a
        bare `.__name__` silently buckets both under "unknown" -- which hides
        exactly the nodes whose sharing is worth reporting.
        """
        parameters = getattr(node, "parameters", None) or {}
        kind = parameters.get("_node_kind")
        if kind == "task":
            return "task"
        operation = getattr(node, "operation", None)
        if operation is None:
            return "leaf" if not getattr(node, "inputs", None) else "unknown"
        name = getattr(operation, "__name__", None)
        if name:
            return name
        return type(operation).__name__

    def _get_nodes_from_dags(
        self, dags: List[RepresentationDag]
    ) -> List[RepresentationNode]:
        nodes = []
        dag_node_slots = 0
        slot_counts = defaultdict(int)
        node_operations = {}
        for dag in dags:
            self.roots.add(dag.root_node_id)
            for node in dag.nodes:
                dag_node_slots += 1
                slot_counts[node.node_id] += 1
                node_operations[node.node_id] = self._operation_label(node)
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

        saved_by_operation = defaultdict(int)
        for node_id, count in slot_counts.items():
            if count > 1:
                saved_by_operation[node_operations.get(node_id, "unknown")] += count - 1
        self.reuse_stats = {
            "n_dags": len(dags),
            "dag_node_slots": dag_node_slots,
            "unique_nodes": len(self.mapping),
            "nodes_saved_by_merging": dag_node_slots - len(self.mapping),
            "shared_nodes": sum(
                1
                for node_id in self.mapping
                if len(self.children.get(node_id, set())) > 1
            ),
            "max_fanout": max(
                (len(children) for children in self.children.values()), default=0
            ),
            "saved_by_operation": dict(saved_by_operation),
            "n_leaves": len(self.leaves),
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
                input_stats = (
                    parent_stats[0] if len(parent_stats) == 1 else parent_stats
                )
                if node not in self.roots:
                    operation = self.mapping[node].operation(
                        params=self.mapping[node].parameters
                    )
                    peak_memory = operation.estimate_peak_memory_bytes(input_stats)
                    input_stats_for_overhead = self._stats_for_overhead(input_stats)
                    peak_memory["cpu_peak_bytes"] += (
                        64 * 1024 + 512 * input_stats_for_overhead.num_instances
                    )  # Placeholder for transformed modality creation overhead
                    peak_memory["cpu_peak_bytes"] *= 1
                    output_stats = self._resolve_output_stats(
                        operation.get_output_stats(input_stats), input_stats
                    )
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
    def _stats_for_overhead(input_stats: Any) -> Any:
        if isinstance(input_stats, list) and len(input_stats) > 0:
            return input_stats[0]
        return input_stats

    @staticmethod
    def _stats_to_bytes(stats: Optional[Any], dtype_size: Optional[int] = None) -> int:
        if stats is None:
            return 0
        if dtype_size is None:
            dtype_size = stats_itemsize(stats)
        num_instances = int(getattr(stats, "num_instances", 0))
        output_shape = tuple(getattr(stats, "output_shape", ()))
        numel = 1
        for dim in output_shape:
            try:
                numel *= int(dim)
            except Exception:
                numel *= 1
        return max(0, int(num_instances * numel * dtype_size))

    @staticmethod
    def _resolve_output_stats(output_stats: Any, input_stats: Any) -> Any:
        if output_stats is None or getattr(output_stats, "dtype", None) is not None:
            return output_stats

        sources = input_stats if isinstance(input_stats, list) else [input_stats]
        sources = [s for s in sources if s is not None]
        sources = [s for s in sources if stats_itemsize(s) > 0]
        if not sources:
            return output_stats

        widest = max(sources, key=lambda s: stats_itemsize(s))
        try:
            output_stats.dtype = stats_dtype(widest)
        except AttributeError:
            pass
        return output_stats
