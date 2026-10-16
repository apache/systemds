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
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import numpy as np

from systemds.scuro.dataloader.base_loader import BaseLoader, LazyFileSequence
import cv2
from systemds.scuro.modality.type import ModalityType


@dataclass
class VideoStats:
    fps: int
    max_length: int
    avg_length: float
    max_width: int
    max_height: int
    max_channels: int
    num_instances: int
    num_total_instances: int
    shape_variance: float = 0.0
    lengths: Optional[Tuple[int, ...]] = None

    @property
    def output_shape(self):
        """
        Approximate output shape for raw video tensors.

        This is used by generic resource estimation logic which expects
        a stats object to expose an ``output_shape`` iterable describing
        the per-instance tensor shape. For videos we approximate this as
        (max_length, max_height, max_width, max_num_channels).
        """
        return (self.max_length, self.max_height, self.max_width, self.max_channels)

    @property
    def avg_output_shape(self):
        return (
            max(1, int(round(self.avg_length))),
            self.max_height,
            self.max_width,
            self.max_channels,
        )


class VideoLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        data_type: Union[np.dtype, str] = np.float16,
        chunk_size: Optional[int] = None,
        load=True,
        fps=None,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            source_path, indices, data_type, chunk_size, ModalityType.VIDEO
        )
        self.load_data_from_file = load
        self.fps = fps
        self.target_size = tuple(int(v) for v in target_size) if target_size else None
        self._all_metadata = []
        files = self.get_file_names(self.indices)
        self.stats, self._all_metadata = self.cached_stats(
            files,
            {
                "dtype": str(data_type),
                "fps": fps,
                "target_size": self.target_size,
                "frame_count": "decodable",
            },
            lambda: (self.get_stats(source_path), self._all_metadata),
        )
        self.stats.lengths = tuple(int(md["length"]) for md in self._all_metadata)

    def load(self):
        if self.chunk_size:
            return super().load()

        self.data = LazyFileSequence(
            self.get_file_names(self.indices), self._decode_data
        )
        self.metadata = self._all_metadata.copy()
        return self.data, self.metadata

    def _decode_data(self, file: str):
        return self._decode_file(file)[0]

    def _frame_interval(self, source_fps: float) -> int:
        if self.fps and source_fps and self.fps < source_fps:
            return max(1, int(round(source_fps / self.fps)))
        return 1

    @staticmethod
    def _decodable_frame_count(cap) -> int:
        count = 0
        while cap.grab():
            count += 1
        return count

    def _stored_length(self, source_length: int, source_fps: float) -> int:
        interval = self._frame_interval(source_fps)
        return int(math.ceil(source_length / interval)) if source_length > 0 else 0

    def _stored_frame_size(self, width: int, height: int) -> Tuple[int, int]:
        return self.target_size if self.target_size else (width, height)

    def _fit_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.target_size is None:
            return frame

        target_w, target_h = self.target_size
        height, width = frame.shape[:2]
        if (width, height) == (target_w, target_h):
            return frame

        scale = max(target_w / width, target_h / height)
        new_w = max(target_w, int(round(width * scale)))
        new_h = max(target_h, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return frame[top : top + target_h, left : left + target_w]

    def _decode_file(self, file: str):
        self.file_sanity_check(file)
        cap = cv2.VideoCapture(file)

        if not cap.isOpened():
            raise ValueError(f"Could not read video at path: {file}")

        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = self._frame_interval(source_fps)
            stored_fps = source_fps / frame_interval if source_fps else source_fps

            scale_denominator = np.dtype(self._data_type).type(255.0)
            frames = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % frame_interval == 0:
                    frame = self._fit_frame(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(self._data_type) / scale_denominator
                    frames.append(frame)
                idx += 1
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames could be decoded from {file}")

        data = np.stack(frames)

        num_frames, height, width = data.shape[0], data.shape[1], data.shape[2]
        metadata = self.modality_type.create_metadata(
            stored_fps, num_frames, width, height, data.shape[3]
        )
        return data, metadata

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        data, metadata = self._decode_file(file)
        self.metadata.append(metadata)
        self.data.append(data)

    def get_stats(self, source_path: str):
        self._all_metadata = []
        self.file_sanity_check(source_path)
        max_length = 0
        max_width = 0
        max_height = 0
        max_num_channels = 0
        num_instances = 0
        stored_lengths = []
        stored_fps = []

        for file in self.get_file_names(self.indices):
            self.file_sanity_check(file)
            cap = cv2.VideoCapture(file)
            if not cap.isOpened():
                raise ValueError(f"Could not read video at path: {file}")
            try:
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                source_length = self._decodable_frame_count(cap)
            finally:
                cap.release()

            length = self._stored_length(source_length, source_fps)
            width, height = self._stored_frame_size(width, height)
            num_channels = 3
            stored_frequency = (
                source_fps / self._frame_interval(source_fps) if source_fps else 0
            )
            self._all_metadata.append(
                self.modality_type.create_metadata(
                    stored_frequency, length, width, height, num_channels
                )
            )

            max_length = max(max_length, length)
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            max_num_channels = max(max_num_channels, num_channels)
            stored_lengths.append(length)
            if source_fps:
                stored_fps.append(stored_frequency)
            num_instances += 1

        num_total_instances = num_instances
        avg_length = float(np.mean(stored_lengths)) if stored_lengths else 0.0
        shape_variance = (
            float(np.std(stored_lengths) / avg_length)
            if stored_lengths and avg_length > 0
            else 0.0
        )
        num_instances = (
            min(num_instances, self.chunk_size)
            if self.chunk_size is not None
            else num_instances
        )
        return VideoStats(
            float(np.mean(stored_fps)) if stored_fps else 0,
            max_length,
            avg_length,
            max_width,
            max_height,
            max_num_channels,
            num_instances,
            num_total_instances,
            shape_variance,
        )

    def estimate_peak_memory_bytes(self) -> dict:
        stats = self.stats
        n = self.chunk_size if self.chunk_size is not None else 1
        n = min(n, stats.num_total_instances)
        per_instance = int(np.prod(stats.avg_output_shape))
        itemsize = np.dtype(self._data_type).itemsize
        return {
            "cpu_peak_bytes": int(n * per_instance * itemsize),
            "gpu_peak_bytes": 0,
        }
