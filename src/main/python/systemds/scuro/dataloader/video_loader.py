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
from typing import List, Optional, Union

import numpy as np

from systemds.scuro.dataloader.base_loader import BaseLoader
import cv2
from systemds.scuro.modality.type import ModalityType


class VideoLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        data_type: Union[np.dtype, str] = np.float16,
        chunk_size: Optional[int] = None,
        load=True,
    ):
        super().__init__(
            source_path, indices, data_type, chunk_size, ModalityType.VIDEO
        )
        self.load_data_from_file = load

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)
        # if not self.load_data_from_file:
        #     self.metadata[file] = self.modality_type.create_video_metadata(
        #         30, 10, 100, 100, 3
        #     )
        # else:
        cap = cv2.VideoCapture(file)

        if not cap.isOpened():
            raise f"Could not read video at path: {file}"

        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_channels = 3

        self.metadata[file] = self.modality_type.create_video_metadata(
            fps, length, width, height, num_channels
        )

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(self._data_type) / 255.0

            frames.append(frame)

        self.data.append(np.stack(frames))
