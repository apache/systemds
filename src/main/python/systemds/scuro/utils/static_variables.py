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
import numpy as np
import torch
from typing import Tuple

PY_LIST_HEADER_BYTES = 56
PY_LIST_SLOT_BYTES = 8
NP_ARRAY_HEADER_BYTES = 112

DEBUG = False

global_rng = np.random.default_rng(42)


def get_seed():
    return global_rng.integers(0, 1024)


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
    if not torch.cuda.is_available():
        return torch.device("cpu")

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

    if device.type == "cpu":
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
