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
import os
from typing import List, Optional, Union

import numpy as np

from systemds.scuro.dataloader.base_loader import BaseLoader
import cv2
from systemds.scuro.modality.type import ModalityType


@dataclass
class VideoStats:
    fps: int
    max_length: int
    max_width: int
    max_height: int
    max_channels: int
    num_instances: int

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


class VideoLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        data_type: Union[np.dtype, str] = np.float16,
        chunk_size: Optional[int] = None,
        load=True,
        fps=None,
    ):
        super().__init__(
            source_path, indices, data_type, chunk_size, ModalityType.VIDEO
        )
        self.load_data_from_file = load
        self.fps = fps
        self.stats = self.get_stats(source_path)

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)
        cap = cv2.VideoCapture(file)

        if not cap.isOpened():
            raise f"Could not read video at path: {file}"

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1
        if self.fps is not None and self.fps < orig_fps:
            frame_interval = int(round(orig_fps / self.fps))
        else:
            self.fps = orig_fps

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_channels = 3

        self.metadata[file] = self.modality_type.create_metadata(
            self.fps, length, width, height, num_channels
        )

        frames = []
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            if idx % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(self._data_type, copy=False) / 255.0
                frames.append(frame)
            idx += 1

        self.data.append(np.stack(frames))

    def get_stats(self, source_path: str):
        self.file_sanity_check(source_path)
        fps = 0
        max_length = 0
        max_width = 0
        max_height = 0
        max_num_channels = 0
        num_instances = 0
        for file in os.listdir(source_path):
            file_name = file.split(".")[0]
            if file_name not in self.indices:
                continue
            self.file_sanity_check(source_path + file)
            cap = cv2.VideoCapture(source_path + file)

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_channels = 3
            max_length = max(max_length, length)
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            max_num_channels = max(max_num_channels, num_channels)
            num_instances += 1
        return VideoStats(
            fps, max_length, max_width, max_height, max_num_channels, num_instances
        )

    def estimate_peak_memory_bytes(self) -> dict:
        s = self.stats
        if self.chunk_size is not None:
            n = self.chunk_size
        else:
            n = s.num_instances
        return {
            "cpu_peak_bytes": n
            * s.output_shape[0]
            * s.output_shape[1]
            * s.output_shape[2]
            * s.output_shape[3]
            * 4,
            "gpu_peak_bytes": 0,
        }
