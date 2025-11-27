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


class ImageLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        data_type: Union[np.dtype, str] = np.float16,
        chunk_size: Optional[int] = None,
        load=True,
        ext=".jpg"
    ):
        super().__init__(
            source_path, indices, data_type, chunk_size, ModalityType.IMAGE, ext
        )
        self.load_data_from_file = load

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)

        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.ndim == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape

        image = image.astype(np.float32) / 255.0

        self.metadata[file] = self.modality_type.create_metadata(
            width, height, channels
        )

        self.data.append(image)