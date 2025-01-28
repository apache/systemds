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
from systemds.scuro.utils.schema_helpers import create_timestamps
import cv2


class VideoLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        chunk_size: Optional[int] = None,
    ):
        super().__init__(source_path, indices, chunk_size)

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)
        cap = cv2.VideoCapture(file)

        if not cap.isOpened():
            raise f"Could not read video at path: {file}"

        self.metadata[file] = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "length": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "num_channels": 3,
        }

        self.metadata[file]["timestamp"] = create_timestamps(
            self.metadata[file]["fps"], self.metadata[file]["length"]
        )

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

        self.data.append(frames)
