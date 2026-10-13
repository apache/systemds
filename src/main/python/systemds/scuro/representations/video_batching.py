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
import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from systemds.scuro.representations.utils import (
    LengthBucketBatchSampler,
    move_batch_to_device,
    pin_memory_for,
)

MIB = 1024 * 1024


@dataclass(frozen=True)
class VideoMemoryProfile:
    clip_frame_bytes: float
    batch_frame_bytes: float
    clip_bytes: float = 0.0
    cpu_frame_bytes: float = 0.0

    def gpu_bytes(self, clip_length: int, batch_frames: int, clips: int) -> float:
        return (
            self.clip_frame_bytes * clip_length
            + self.batch_frame_bytes * batch_frames
            + self.clip_bytes * clips
        )


class FrameBudgetedVideoModel:
    _BATCH_ACTIVATION_BUDGET_BYTES = 10 * 1024 * MIB
    _ALLOCATOR_OVERHEAD = 1.65
    _GPU_WORKSPACE_BYTES = 2048 * MIB
    _SAFETY_MARGIN_BYTES = 100 * MIB
    _GPU_MEMORY_FRACTION = 0.9
    _MODEL_INPUT_FRAME_BYTES = 3 * 224 * 224 * 4

    _max_batch_frames: Optional[int] = None

    def _memory_profile(self) -> VideoMemoryProfile:
        raise NotImplementedError

    def _model_lengths(self, lengths: Sequence[int]) -> List[int]:
        """Frames per clip as the model receives them."""
        return [max(int(length), 1) for length in lengths]

    @property
    def max_batch_frames(self) -> int:
        if self._max_batch_frames is not None:
            return self._max_batch_frames
        return max(
            1,
            int(
                self._BATCH_ACTIVATION_BUDGET_BYTES
                // self._memory_profile().batch_frame_bytes
            ),
        )

    @max_batch_frames.setter
    def max_batch_frames(self, value):
        self._max_batch_frames = None if value is None else max(1, int(value))

    def _release_cache_on_new_shape(self, frames) -> None:
        if not isinstance(frames, torch.Tensor):
            # Clips of differing frame sizes stay a list until preprocessing.
            return
        shape = tuple(frames.shape)
        if frames.is_cuda and shape != getattr(self, "_last_batch_shape", None):
            if getattr(self, "_last_batch_shape", None) is not None:
                torch.cuda.empty_cache()
            self._last_batch_shape = shape

    def _longest_clip(self, input_stats) -> int:
        max_length = int(getattr(input_stats, "max_length", 1) or 1)
        return self._model_lengths([max_length])[0]

    def gpu_memory_limit(self) -> Optional[int]:
        if not torch.cuda.is_available():
            return None
        total = min(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        )
        return int(total * self._GPU_MEMORY_FRACTION)

    def _batch_gpu_bytes(self, clip_length: int, clips: int) -> float:
        activations = self._memory_profile().gpu_bytes(
            clip_length, clip_length * clips, clips
        )
        return (
            self._model_bytes()
            + activations * self._ALLOCATOR_OVERHEAD
            + self._GPU_WORKSPACE_BYTES
        )

    def _clips_fitting_gpu(self, clip_length: int, clips: int, limit) -> int:
        if limit is None:
            return clips
        while clips > 0 and self._batch_gpu_bytes(clip_length, clips) > limit:
            clips -= 1
        return clips

    def _plan_batches(self, lengths, batch_size=None, limit=None):
        lengths = self._model_lengths(lengths)
        sampler = LengthBucketBatchSampler(
            lengths,
            self.batch_size if batch_size is None else batch_size,
            exact=True,
            max_batch_items=self.max_batch_frames,
        )
        on_gpu, on_cpu = [], []
        for batch in sampler:
            fitting = self._clips_fitting_gpu(lengths[batch[0]], len(batch), limit)
            if fitting == 0:
                on_cpu.extend((tuple([position]), False) for position in batch)
                continue
            on_gpu.extend(
                (tuple(batch[start : start + fitting]), True)
                for start in range(0, len(batch), fitting)
            )
        return on_gpu + on_cpu

    def _device_limit(self):
        device = getattr(self, "device", None)
        if device is None or torch.device(device).type != "cuda":
            return None
        return self.gpu_memory_limit()

    def _cpu_model(self):
        if torch.device(self.device).type == "cpu":
            return self.model
        return copy.deepcopy(self.model).to("cpu")

    def _run_batches(self, dataset, lengths, forward, collate_fn=None) -> None:
        plan = self._plan_batches(lengths, limit=self._device_limit())
        on_device = [list(batch) for batch, gpu in plan if gpu]
        on_cpu = [list(batch) for batch, gpu in plan if not gpu]
        loader_args = {} if collate_fn is None else {"collate_fn": collate_fn}
        failed = []
        if on_device:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=on_device,
                pin_memory=pin_memory_for(self.device),
                **loader_args,
            )
            for batch_positions, batch in zip(on_device, loader):
                out_of_memory = False
                try:
                    device_batch = move_batch_to_device(batch, self.device)
                    self._release_cache_on_new_shape(device_batch["data"])
                    forward(self.model, device_batch)
                except torch.cuda.OutOfMemoryError:
                    out_of_memory = True
                if out_of_memory:
                    device_batch = None
                    torch.cuda.empty_cache()
                    self._last_batch_shape = None
                    failed.append(batch_positions)
        on_cpu = failed + on_cpu
        if on_cpu:
            model = self._cpu_model()
            loader = torch.utils.data.DataLoader(
                dataset, batch_sampler=on_cpu, **loader_args
            )
            for batch in loader:
                forward(model, batch)

    def _peak_batch_frames(self, input_stats) -> int:
        lengths = getattr(input_stats, "lengths", None)
        if lengths:
            model_lengths = self._model_lengths(lengths)
            return max(
                model_lengths[batch[0]] * len(batch)
                for batch, _ in self._plan_batches(lengths)
            )
        longest = self._longest_clip(input_stats)
        return max(longest, min(self.batch_size * longest, self.max_batch_frames))

    def _planned_peaks(self, input_stats, limit):
        lengths = getattr(input_stats, "lengths", None)
        if lengths:
            model_lengths = self._model_lengths(lengths)
            gpu, cpu_frames = 0.0, 0
            for batch, on_gpu in self._plan_batches(lengths, limit=limit):
                length = model_lengths[batch[0]]
                if on_gpu:
                    gpu = max(gpu, self._batch_gpu_bytes(length, len(batch)))
                else:
                    cpu_frames = max(cpu_frames, length)
            return gpu, cpu_frames
        longest = self._longest_clip(input_stats)
        gpu = (
            self._model_bytes()
            + self._memory_profile().gpu_bytes(
                longest,
                self._peak_batch_frames(input_stats),
                min(self.batch_size, self.max_batch_frames),
            )
            * self._ALLOCATOR_OVERHEAD
            + self._GPU_WORKSPACE_BYTES
        )
        cpu_frames = 0
        if limit is not None:
            gpu = min(gpu, limit)
            if self._clips_fitting_gpu(longest, 1, limit) == 0:
                cpu_frames = longest
        return gpu, cpu_frames

    def _model_bytes(self) -> int:
        return sum(
            t.nelement() * t.element_size()
            for t in (*self.model.parameters(), *self.model.buffers())
        )

    @staticmethod
    def _raw_frame_bytes(input_stats) -> int:
        raw_dtype = getattr(input_stats, "dtype", None)
        try:
            itemsize = np.dtype(np.float32 if raw_dtype is None else raw_dtype).itemsize
        except TypeError:
            itemsize = np.dtype(np.float32).itemsize
        return (
            max(int(getattr(input_stats, "max_height", 224) or 224), 1)
            * max(int(getattr(input_stats, "max_width", 224) or 224), 1)
            * max(int(getattr(input_stats, "max_channels", 3) or 3), 1)
            * itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        frames = self._peak_batch_frames(input_stats)
        model_bytes = self._model_bytes()
        gpu_peak, cpu_frames = self._planned_peaks(input_stats, self.gpu_memory_limit())
        cpu_peak = (
            model_bytes
            + frames
            * (self._raw_frame_bytes(input_stats) + 3 * self._MODEL_INPUT_FRAME_BYTES)
            + self.estimate_output_memory_bytes(input_stats)
            + self._SAFETY_MARGIN_BYTES
        )
        if cpu_frames:
            cpu_peak += (
                model_bytes + cpu_frames * self._memory_profile().cpu_frame_bytes
            )
        return {"cpu_peak_bytes": int(cpu_peak), "gpu_peak_bytes": int(gpu_peak)}

    def _distinct_batch_sizes(self, values, lengths, groups, limit):
        seen = set()
        kept = []
        for value in values:
            key = tuple(
                tuple(
                    (tuple(group[i] for i in batch), on_gpu)
                    for batch, on_gpu in self._plan_batches(
                        [lengths[i] for i in group], int(value), limit
                    )
                )
                for group in groups
            )
            if key not in seen:
                seen.add(key)
                kept.append(value)
        return kept

    def filter_parameter_domain(self, name, values, input_stats):
        lengths = getattr(input_stats, "lengths", None)
        if name != "batch_size" or not lengths:
            return values
        everything = list(range(len(lengths)))
        groups_per_grouping = [[everything]]
        per_chunk = int(getattr(input_stats, "num_instances", 0) or 0)
        if 0 < per_chunk < len(lengths):
            groups_per_grouping.append(
                [
                    everything[start : start + per_chunk]
                    for start in range(0, len(lengths), per_chunk)
                ]
            )
        groups = [group for grouping in groups_per_grouping for group in grouping]
        return self._distinct_batch_sizes(
            values, lengths, groups, self.gpu_memory_limit()
        )
