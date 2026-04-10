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
from typing import List, Tuple, Optional, Dict, Any
import torch
import psutil

from systemds.scuro.drsearch.representation_dag import RepresentationDag
from systemds.scuro.modality.modality import Modality


def get_peak_memory_from_dag_group(
    dag_group: List[RepresentationDag], modality: Modality
) -> tuple[float, float]:
    peak_memory_cpu = 0.0
    peak_memory_gpu = 0.0
    leaf_memory_bytes = modality.estimate_memory_bytes()
    for dag in dag_group:
        prev_stats = modality.get_stats()
        for node in dag.nodes[1:]:
            peak_memory = node.operation().estimate_peak_memory_bytes(prev_stats)
            peak_memory_cpu = max(peak_memory_cpu, peak_memory["cpu_peak_bytes"])
            peak_memory_gpu = max(peak_memory_gpu, peak_memory["gpu_peak_bytes"])
            prev_stats = node.operation().get_output_shape(prev_stats)

    return (peak_memory_cpu + leaf_memory_bytes, peak_memory_gpu)


class DAGGroupScheduler:
    def __init__(
        self,
        cpu_margin: float = 0.8,
        gpu_margin: float = 0.8,
        shared_state: Optional[Dict[str, Any]] = None,
        lock=None,
        dag_groups: List[List[RepresentationDag]] = None,
        modality: Modality = None,
    ):
        self._margin = (cpu_margin, gpu_margin)
        self._n_gpu = (
            torch.cuda.device_count() if torch and torch.cuda.is_available() else 0
        )
        self._shared = shared_state
        self._lock = lock
        if self._shared is None:
            self._shared = {
                "cpu_in_use": 0.0,
                "gpu_in_use": {i: 0.0 for i in range(max(1, self._n_gpu))},
            }
        else:
            self._shared.setdefault("cpu_in_use", 0.0)
            self._shared.setdefault("gpu_in_use", {})
        self.group_resources = []
        for dag_group in dag_groups:
            cpu_mem, gpu_mem = get_peak_memory_from_dag_group(dag_group, modality)
            self.group_resources.append((cpu_mem, gpu_mem))

    def _avail_cpu(self) -> float:
        available_memory = (psutil.virtual_memory().available) if psutil else 4096.0
        return max(
            0.0,
            available_memory * self._margin[0] - float(self._shared["cpu_in_use"]),
        )

    def _avail_gpu(self, i: int) -> float:
        available_gpu_memory = 0.0
        try:
            free, _ = torch.cuda.mem_get_info(i)
            available_gpu_memory = free
        except Exception:
            pass
        g = self._shared["gpu_in_use"]
        in_use = g.get(str(i), g.get(i, 0.0))
        return max(0.0, available_gpu_memory * self._margin[1] - in_use)

    def _update(
        self, cpu_delta: float, gpu_delta: float, gpu_id: Optional[int]
    ) -> None:
        self._shared["cpu_in_use"] = max(0.0, self._shared["cpu_in_use"] + cpu_delta)
        if gpu_id is not None and gpu_delta != 0:
            g = self._shared["gpu_in_use"]
            k = str(gpu_id)
            g[k] = max(0.0, g.get(k, 0) + gpu_delta)

    def reserve(self, cpu_mb: float, gpu_mb: float) -> Tuple[bool, Optional[int]]:
        if self._lock:
            self._lock.acquire()
        try:
            if self._avail_cpu() < cpu_mb:
                return (False, None)
            if gpu_mb <= 0:
                self._update(cpu_mb, 0, None)
                return (True, None)
            if self._n_gpu == 0 and self._avail_cpu() >= gpu_mb + cpu_mb:
                self._update(cpu_mb + gpu_mb, 0, None)
                return (True, None)

            for i in range(self._n_gpu):
                if i is not None and self._avail_gpu(i) >= gpu_mb:
                    self._update(cpu_mb, gpu_mb, i)
                    return (True, i)

            return (False, None)
        finally:
            if self._lock:
                self._lock.release()

    def release(self, cpu_mb: float, gpu_mb: float, gpu_id: Optional[int]) -> None:
        if self._lock:
            self._lock.acquire()
        try:
            if self._n_gpu == 0:
                self._update(-cpu_mb - gpu_mb, 0, None)
                return
            if gpu_id is not None:
                self._update(-cpu_mb, -gpu_mb, gpu_id)
            else:
                self._update(-cpu_mb - gpu_mb, 0, None)
        finally:
            if self._lock:
                self._lock.release()

    def get_runnable(
        self,
        pending_resources: List[Tuple[int, float, float, Optional[int]]],
        max_concurrent: Optional[int] = None,
        largest_first: bool = True,
    ) -> List[Tuple[int, Optional[int]]]:
        order = sorted(
            range(len(pending_resources)),
            key=lambda i: pending_resources[i][1] + pending_resources[i][2],
            reverse=largest_first,
        )
        runnable = []
        for i in order:
            if max_concurrent is not None and len(runnable) >= max_concurrent:
                break
            idx, cpu_bytes, gpu_bytes = pending_resources[i]
            ok, assigned = self.reserve(cpu_bytes, gpu_bytes)
            if ok:
                runnable.append((idx, assigned))
        return runnable
