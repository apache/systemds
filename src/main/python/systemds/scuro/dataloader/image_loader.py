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
class ImageStats:
    max_width: int
    max_height: int
    max_channels: int
    num_instances: int
    output_shape: tuple
    average_width: int
    average_height: int
    average_channels: int


class ImageLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        data_type: Union[np.dtype, str] = np.float16,
        chunk_size: Optional[int] = None,
        load=True,
        ext=".jpg",
    ):
        super().__init__(
            source_path, indices, data_type, chunk_size, ModalityType.IMAGE, ext
        )
        self.load_data_from_file = load
        self.stats = self.get_stats(source_path)

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)

        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.ndim == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape

        image = image.astype(np.uint8, copy=False)

        self.metadata.append(
            self.modality_type.create_metadata(width, height, channels)
        )

        self.data.append(image)

    def get_stats(self, source_path: str):
        max_width = 0
        max_height = 0
        max_channels = 0
        num_instances = 0
        average_width = 0
        average_height = 0
        average_channels = 0
        for file in self.indices:
            path = os.path.join(source_path, f"{file}{self._ext}")
            # if self.chunk_size is None:
            #     self.extract(path)
            #     md = self.metadata[path]
            #     max_width = max(max_width, md["width"])
            #     max_height = max(max_height, md["height"])
            #     max_channels = max(max_channels, md["num_channels"])
            #     num_instances += 1
            # else:
            self.file_sanity_check(path)
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = image.shape
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            max_channels = max(max_channels, channels)
            num_instances += 1
            average_width += width
            average_height += height
            average_channels += channels
        average_width = average_width / num_instances
        average_height = average_height / num_instances
        average_channels = average_channels / num_instances
        return ImageStats(
            max_width,
            max_height,
            max_channels,
            num_instances,
            (average_width, average_height, average_channels),
            average_width,
            average_height,
            average_channels,
        )
