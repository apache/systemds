from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import psutil
import torch


class NodeResourceScheduler:
    def __init__(
        self,
        cpu_margin: float = 0.8,
        gpu_margin: float = 0.8,
        shared_state: Optional[Dict[str, Any]] = None,
        lock=None,
    ):
        self._margin = (cpu_margin, gpu_margin)
        self._n_gpu = (
            torch.cuda.device_count() if torch and torch.cuda.is_available() else 0
        )
        self._shared = shared_state
        self._lock = lock
        if self._shared is None:
            self._shared = {
                "cpu_exec_in_use": 0.0,
                "cpu_cache_in_use": 0.0,
                "gpu_exec_in_use": {i: 0.0 for i in range(max(1, self._n_gpu))},
                "gpu_cache_in_use": {i: 0.0 for i in range(max(1, self._n_gpu))},
            }
        else:
            self._shared.setdefault("cpu_exec_in_use", 0.0)
            self._shared.setdefault("cpu_cache_in_use", 0.0)
            self._shared.setdefault("gpu_exec_in_use", {})
            self._shared.setdefault("gpu_cache_in_use", {})

    def _avail_cpu(self) -> float:
        available_memory = psutil.virtual_memory().available if psutil else 4096.0
        in_use = float(self._shared["cpu_exec_in_use"]) + float(
            self._shared["cpu_cache_in_use"]
        )
        return max(0.0, available_memory * self._margin[0] - in_use)

    def _avail_gpu(self, i: int) -> float:
        available_gpu_memory = 0.0
        try:
            free, _ = torch.cuda.mem_get_info(i)
            available_gpu_memory = free
        except Exception:
            pass
        g_exec = self._shared["gpu_exec_in_use"]
        g_cache = self._shared["gpu_cache_in_use"]
        in_use = g_exec.get(str(i), g_exec.get(i, 0.0)) + g_cache.get(
            str(i), g_cache.get(i, 0.0)
        )
        return max(0.0, available_gpu_memory * self._margin[1] - in_use)

    def _update_exec(
        self, cpu_delta: float, gpu_delta: float, gpu_id: Optional[int]
    ) -> None:
        self._shared["cpu_exec_in_use"] = max(
            0.0, self._shared["cpu_exec_in_use"] + cpu_delta
        )
        if gpu_id is not None and gpu_delta != 0:
            g = self._shared["gpu_exec_in_use"]
            k = str(gpu_id)
            g[k] = max(0.0, g.get(k, 0) + gpu_delta)

    def _update_cache(
        self, cpu_delta: float, gpu_delta: float, gpu_id: Optional[int]
    ) -> None:
        self._shared["cpu_cache_in_use"] = max(
            0.0, self._shared["cpu_cache_in_use"] + cpu_delta
        )
        if gpu_id is not None and gpu_delta != 0:
            g = self._shared["gpu_cache_in_use"]
            k = str(gpu_id)
            g[k] = max(0.0, g.get(k, 0) + gpu_delta)

    def reserve_exec(
        self, cpu_bytes: float, gpu_bytes: float
    ) -> Tuple[bool, Optional[int]]:
        if self._lock:
            self._lock.acquire()
        try:
            if self._avail_cpu() < cpu_bytes:
                return (False, None)
            if gpu_bytes <= 0:
                self._update_exec(cpu_bytes, 0.0, None)
                return (True, None)

            if self._n_gpu == 0 and self._avail_cpu() >= gpu_bytes + cpu_bytes:
                self._update_exec(cpu_bytes + gpu_bytes, 0.0, None)
                return (True, None)

            for i in range(self._n_gpu):
                if self._avail_gpu(i) >= gpu_bytes:
                    self._update_exec(cpu_bytes, gpu_bytes, i)
                    return (True, i)
            return (False, None)
        finally:
            if self._lock:
                self._lock.release()

    def can_execute_now(
        self, cpu_bytes: float, gpu_bytes: float, gpu_id: Optional[int]
    ) -> bool:
        if self._lock:
            self._lock.acquire()
        try:
            cpu_available = psutil.virtual_memory().available if psutil else 4096.0
            if cpu_available * self._margin[0] < cpu_bytes:
                return False

            if gpu_bytes <= 0:
                return True

            if self._n_gpu == 0:
                return cpu_available * self._margin[0] >= (cpu_bytes + gpu_bytes)

            if gpu_id is None or gpu_id >= self._n_gpu:
                return False

            try:
                free_gpu, _ = torch.cuda.mem_get_info(gpu_id)
                return free_gpu * self._margin[1] >= gpu_bytes
            except Exception:
                return False
        finally:
            if self._lock:
                self._lock.release()

    def release_exec(
        self, cpu_bytes: float, gpu_bytes: float, gpu_id: Optional[int]
    ) -> None:
        if self._lock:
            self._lock.acquire()
        try:
            if self._n_gpu == 0:
                self._update_exec(-cpu_bytes - gpu_bytes, 0.0, None)
                return
            if gpu_id is not None:
                self._update_exec(-cpu_bytes, -gpu_bytes, gpu_id)
            else:
                self._update_exec(-cpu_bytes - gpu_bytes, 0.0, None)
        finally:
            if self._lock:
                self._lock.release()

    def reserve_cache(
        self, cpu_bytes: float, gpu_bytes: float, gpu_id: Optional[int]
    ) -> bool:
        if self._lock:
            self._lock.acquire()
        try:
            if self._avail_cpu() < cpu_bytes:
                return False

            if gpu_bytes <= 0:
                self._update_cache(cpu_bytes, 0.0, None)
                return True

            if self._n_gpu == 0 and self._avail_cpu() >= gpu_bytes + cpu_bytes:
                self._update_cache(cpu_bytes + gpu_bytes, 0.0, None)
                return True

            if gpu_id is not None and gpu_id < self._n_gpu:
                if self._avail_gpu(gpu_id) >= gpu_bytes:
                    self._update_cache(cpu_bytes, gpu_bytes, gpu_id)
                    return True
                return False

            for i in range(self._n_gpu):
                if self._avail_gpu(i) >= gpu_bytes:
                    self._update_cache(cpu_bytes, gpu_bytes, i)
                    return True
            return False
        finally:
            if self._lock:
                self._lock.release()

    def release_cache(
        self, cpu_bytes: float, gpu_bytes: float, gpu_id: Optional[int]
    ) -> None:
        if self._lock:
            self._lock.acquire()
        try:
            if self._n_gpu == 0:
                self._update_cache(-cpu_bytes - gpu_bytes, 0.0, None)
                return
            if gpu_id is not None:
                self._update_cache(-cpu_bytes, -gpu_bytes, gpu_id)
            else:
                self._update_cache(-cpu_bytes - gpu_bytes, 0.0, None)
        finally:
            if self._lock:
                self._lock.release()


class ExternalRefCountCache:
    def __init__(self, scheduler: NodeResourceScheduler):
        self._scheduler = scheduler
        self._cache: Dict[str, Any] = {}
        self._remaining_children: Dict[str, int] = {}
        self._mem: Dict[str, Tuple[float, float, Optional[int]]] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def put(
        self,
        key: str,
        value: Any,
        remaining_children: int,
        cpu_bytes: float,
        gpu_bytes: float,
        gpu_id: Optional[int],
    ) -> None:
        if remaining_children <= 0:
            return
        if not self._scheduler.reserve_cache(cpu_bytes, gpu_bytes, gpu_id):
            return
        self._cache[key] = value
        self._remaining_children[key] = remaining_children
        self._mem[key] = (cpu_bytes, gpu_bytes, gpu_id)

    def consume(self, key: str) -> None:
        if key not in self._remaining_children:
            return
        self._remaining_children[key] -= 1
        if self._remaining_children[key] <= 0:
            self.evict(key)

    def evict(self, key: str) -> None:
        self._cache.pop(key, None)
        self._remaining_children.pop(key, None)
        mem = self._mem.pop(key, None)
        if mem is not None:
            self._scheduler.release_cache(*mem)
