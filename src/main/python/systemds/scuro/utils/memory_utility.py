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
import ctypes
import os
import re
import resource
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass
import numpy as np
from sympy import Dict
import torch
from typing import List, Optional, Tuple
import psutil
import gc
from systemds.scuro.utils.static_variables import cuda_allowed, deterministic_enabled

_CPU_MEMORY_BUDGET_FRACTION = float(
    os.environ.get("SCURO_CPU_MEMORY_BUDGET_FRACTION", "0.5")
)
_CPU_MEMORY_BUDGET_GB = os.environ.get("SCURO_CPU_MEMORY_BUDGET_GB")


def cpu_memory_budget_bytes(modality_type: Optional[str] = None) -> float:
    """CPU memory budget for one executor run.

    SCURO_CPU_MEMORY_BUDGET_GB_<MODALITY TYPE> (e.g. ..._AUDIO) overrides
    SCURO_CPU_MEMORY_BUDGET_GB for executors over that modality type, so a
    modality whose memory estimates are far below actual use can be throttled
    without slowing down every other modality.
    """
    budget = float(psutil.virtual_memory().available) * _CPU_MEMORY_BUDGET_FRACTION
    limit = _CPU_MEMORY_BUDGET_GB
    if modality_type:
        limit = os.environ.get(
            f"SCURO_CPU_MEMORY_BUDGET_GB_{str(modality_type).upper()}", limit
        )
    if limit:
        budget = min(budget, float(limit) * 1024**3)
    return budget


def get_model_size_mb(model: torch.nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024**2)


def get_gpu_free_memory_mb(device_index: int = 0) -> float:
    free, _ = torch.cuda.mem_get_info(device_index)
    return free / (1024**2)


def get_best_gpu() -> Tuple[int, float]:
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return -1, 0.0

    best_gpu = 0
    best_free = 0.0
    for i in range(num_gpus):
        free_mb = get_gpu_free_memory_mb(i)
        if free_mb > best_free:
            best_free = free_mb
            best_gpu = i

    return best_gpu, best_free


def get_device_for_model(
    model: torch.nn.Module, memory_factor: float = 1.5
) -> torch.device:
    if not cuda_allowed():
        return torch.device("cpu")
    if deterministic_enabled():
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    model_mb = get_model_size_mb(model)
    required_mb = model_mb * memory_factor

    gpu_idx, free_mb = get_best_gpu()

    if free_mb >= required_mb:
        return torch.device(f"cuda:{gpu_idx}")

    return torch.device("cpu")


def estimate_batch_memory_mb(
    sample_data, tokenizer=None, max_seq_length=512, dtype=torch.float32
) -> float:
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()

    if tokenizer is not None:
        num_tokens = max_seq_length
        num_input_tensors = 3
        input_bytes = num_tokens * num_input_tensors * 8  # int64

        activation_bytes = num_tokens * 768 * 4 * bytes_per_element

        return (input_bytes + activation_bytes) / (1024**2)

    elif isinstance(sample_data, torch.Tensor):
        return sample_data.numel() * bytes_per_element / (1024**2)

    elif isinstance(sample_data, np.ndarray):
        return sample_data.nbytes / (1024**2)

    else:
        return 1.0


def compute_batch_size(
    model: torch.nn.Module,
    device: torch.device,
    sample_data,
    tokenizer=None,
    max_seq_length: int = 512,
    dtype=torch.float32,
    min_batch_size: int = 1,
    max_batch_size: int = 128,
    memory_fraction: float = 0.8,
) -> int:

    if device.type == "cpu" or deterministic_enabled():
        return max_batch_size

    gpu_idx = device.index if device.index is not None else 0
    free_mb = get_gpu_free_memory_mb(gpu_idx)
    usable_mb = free_mb * memory_fraction

    per_sample_mb = estimate_batch_memory_mb(
        sample_data, tokenizer=tokenizer, max_seq_length=max_seq_length, dtype=dtype
    )

    if per_sample_mb <= 0:
        return max_batch_size

    computed = int(usable_mb / per_sample_mb)

    batch_size = max(min_batch_size, min(computed, max_batch_size))

    return batch_size


def get_device(gpu_id: int = None):
    if gpu_id is not None:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda" if cuda_allowed() else "cpu")


def get_cpu_memory_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if __import__("sys").platform == "darwin":
        rss /= 1024  # macOS reports bytes
    return rss / 1024  # MB


def get_gpu_memory_mb(device):
    if not torch.cuda.is_available() or device.type == "cpu":
        return 0.0
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / (1024**2)


def gpu_memory_info():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return []
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            infos = []
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                infos.append(
                    dict(index=i, free_b=int(mem.free), total_b=int(mem.total))
                )
            return infos
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    infos = []
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        free_b, total_b = torch.cuda.mem_get_info()
        infos.append(dict(index=i, free_b=free_b, total_b=total_b))
    return infos


def log_memory(operation_name: str, device: torch.device):
    cpu_mb = get_cpu_memory_mb()
    gpu_mb = get_gpu_memory_mb(device)
    current_mb = psutil.Process().memory_info().rss / (1024**2)
    print(
        f"[mem] {operation_name}: CPU={cpu_mb:.1f} MB, GPU={gpu_mb:.1f} MB, Current={current_mb:.1f} MB"
    )


def _shared_container_bytes(data) -> Optional[int]:
    if getattr(data, "shm_name", None) is None:
        return None
    payload = getattr(data, "payload_nbytes", None)  # SharedStringList
    if payload is not None:
        return int(payload)
    dtype = getattr(data, "_dtype", None)
    if dtype is None:
        return 0
    total_elems = getattr(data, "total_elems", None)  # Shared(Grouped)ArrayList
    if total_elems is None:
        total_elems = int(np.prod(getattr(data, "shape", ())))  # SharedNDArray
    return int(total_elems) * int(np.dtype(dtype).itemsize)


def split_numpy_like_bytes(data) -> Tuple[int, int]:
    if data is None:
        return 0, 0
    shared = _shared_container_bytes(data)
    if shared is not None:
        return 0, shared
    if isinstance(data, dict):
        values = data.values()
    elif isinstance(data, (list, tuple)):
        values = data
    else:
        return estimate_numpy_like_bytes(data), 0
    private = shared = 0
    for v in values:
        p, s = split_numpy_like_bytes(v)
        private += p
        shared += s
    return int(private), int(shared)


def estimate_numpy_like_bytes(data) -> int:
    if data is None:
        return 0
    shared = _shared_container_bytes(data)
    if shared is not None:
        return shared
    if isinstance(data, np.ndarray):
        return int(data.nbytes)
    if isinstance(data, torch.Tensor):
        return int(data.numel() * data.element_size())
    if isinstance(data, memoryview):
        return int(data.nbytes)
    if isinstance(data, (bytes, bytearray)):
        return int(len(data))
    if isinstance(data, dict):
        return int(sum(estimate_numpy_like_bytes(v) for v in data.values()))
    if isinstance(data, (list, tuple)):
        return int(sum(estimate_numpy_like_bytes(v) for v in data))
    return int(sys.getsizeof(data))


def estimate_modality_bytes(modality) -> int:
    if modality is None:
        return 0
    data_bytes = estimate_numpy_like_bytes(getattr(modality, "data", None))
    metadata = getattr(modality, "metadata", None)
    metadata_bytes = estimate_numpy_like_bytes(metadata)
    return int(data_bytes + metadata_bytes)


def split_modality_bytes(modality) -> Tuple[int, int]:
    if modality is None:
        return 0, 0
    data_p, data_s = split_numpy_like_bytes(getattr(modality, "data", None))
    meta_p, meta_s = split_numpy_like_bytes(getattr(modality, "metadata", None))
    return int(data_p + meta_p), int(data_s + meta_s)


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "cudamalloc" in msg
        or "cuda error: out of memory" in msg
    )


def cuda_oom_peak_bytes(exc: BaseException) -> int:
    if not torch.cuda.is_available():
        return -1
    peak = torch.cuda.max_memory_allocated()
    m = re.search(r"tried to allocate ([\d.]+) (bytes|kib|mib|gib)", str(exc).lower())
    if m:
        unit = ("bytes", "kib", "mib", "gib").index(m.group(2))
        peak += int(float(m.group(1)) * 1024**unit)
    return int(peak)


def release_freed_heap() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def cleanup_gpu(gpu_id: Optional[int]) -> None:
    if gpu_id is None or not torch.cuda.is_available():
        return
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class MemoryMeasurement:
    increment_bytes: int
    footprint_bytes: int
    input_resident_bytes: int
    traced_peak_bytes: int
    rss_delta_bytes: int
    peak_abs_rss_bytes: int
    input_shared_bytes: int = 0
    shared_rss_delta_bytes: int = 0
    output_shm_bytes: int = 0
    pid: int = -1
    t_start_unix: float = -1.0
    t_end_unix: float = -1.0
    baseline_private_rss_bytes: int = -1
    peak_private_rss_bytes: int = -1


def merge_memory_measurements(
    measurements: List[Optional[MemoryMeasurement]],
) -> Optional[MemoryMeasurement]:
    present = [m for m in measurements if m is not None]
    if not present:
        return None
    return MemoryMeasurement(
        increment_bytes=max(m.increment_bytes for m in present),
        footprint_bytes=max(m.footprint_bytes for m in present),
        input_resident_bytes=max(m.input_resident_bytes for m in present),
        traced_peak_bytes=max(m.traced_peak_bytes for m in present),
        rss_delta_bytes=max(m.rss_delta_bytes for m in present),
        peak_abs_rss_bytes=max(m.peak_abs_rss_bytes for m in present),
        input_shared_bytes=max(m.input_shared_bytes for m in present),
        shared_rss_delta_bytes=max(m.shared_rss_delta_bytes for m in present),
        output_shm_bytes=max(m.output_shm_bytes for m in present),
        pid=present[0].pid,
        t_start_unix=min(m.t_start_unix for m in present),
        t_end_unix=max(m.t_end_unix for m in present),
        baseline_private_rss_bytes=min(m.baseline_private_rss_bytes for m in present),
        peak_private_rss_bytes=max(m.peak_private_rss_bytes for m in present),
    )


def measure_memory_during(
    fn,
    *args,
    input_resident_bytes: int = 0,
    input_shared_bytes: int = 0,
    sample_s: float = 0.01,
    **kwargs,
):
    proc = psutil.Process(os.getpid())

    def _sample():
        mi = proc.memory_info()
        shared = getattr(mi, "shared", 0)
        return int(mi.rss), int(mi.rss - shared), int(shared)

    t_start_unix = time.time()
    baseline_rss, baseline_priv, baseline_shared = _sample()
    peak_rss, peak_priv, peak_shared = baseline_rss, baseline_priv, baseline_shared
    stop = threading.Event()

    def sampler():
        nonlocal peak_rss, peak_priv, peak_shared
        while not stop.is_set():
            rss, priv, shared = _sample()
            peak_rss = max(peak_rss, rss)
            peak_priv = max(peak_priv, priv)
            peak_shared = max(peak_shared, shared)
            time.sleep(sample_s)

    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    traced_baseline, _ = tracemalloc.get_traced_memory()

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        out = fn(*args, **kwargs)
    finally:
        stop.set()
        t.join()
        _, traced_peak = tracemalloc.get_traced_memory()
        if owns_tracing:
            tracemalloc.stop()

    traced_increment = max(int(traced_peak) - int(traced_baseline), 0)
    rss_delta = max(int(peak_priv) - int(baseline_priv), 0)
    increment = max(traced_increment, rss_delta)

    return out, MemoryMeasurement(
        increment_bytes=increment,
        footprint_bytes=increment + int(input_resident_bytes),
        input_resident_bytes=int(input_resident_bytes),
        traced_peak_bytes=traced_increment,
        rss_delta_bytes=rss_delta,
        peak_abs_rss_bytes=int(peak_rss),
        input_shared_bytes=int(input_shared_bytes),
        shared_rss_delta_bytes=max(int(peak_shared) - int(baseline_shared), 0),
        pid=os.getpid(),
        t_start_unix=t_start_unix,
        t_end_unix=time.time(),
        baseline_private_rss_bytes=int(baseline_priv),
        peak_private_rss_bytes=int(peak_priv),
    )
