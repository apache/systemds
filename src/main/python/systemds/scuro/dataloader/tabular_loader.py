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
import numpy as np
from typing import List, Optional, Union

from systemds.scuro.dataloader.base_loader import BaseLoader
from systemds.scuro.modality.type import ModalityType


@dataclass
class TabularStats:
    num_instances: int
    num_features: int
    output_shape: tuple
    output_shape_is_known: bool = True


class TabularLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        feature_names: Optional[List[str]] = None,
        data_type: Union[np.dtype, str] = np.float32,
        chunk_size: Optional[int] = None,
        normalize: bool = False,
        file_format: str = "npy",
        modality_type: Optional[ModalityType] = ModalityType.EMBEDDING,
    ):
        super().__init__(source_path, indices, data_type, chunk_size, modality_type)
        self.feature_names = feature_names
        self.normalize = normalize
        self.file_format = file_format.lower()
        if self.file_format != "npy":
            raise ValueError(f"Unsupported file format: {self.file_format}")
        self.stats = self.get_stats(source_path)

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)
        data = np.load(file).astype(self._data_type, copy=False).reshape(-1)

        if self.normalize:
            mean = np.mean(data)
            std = np.std(data)
            data = (data - mean) / (std + 1e-8)

        self.metadata.append(self.modality_type.create_metadata(data))
        self.data.append(data)

    def get_stats(self, source_path: str) -> TabularStats:
        num_instances = 0
        num_features = 0
        for file_name in self.indices:
            file = source_path + file_name + "." + self.file_format
            self.file_sanity_check(file)
            data = np.load(file)
            num_features = max(num_features, int(np.prod(data.shape)))
            num_instances += 1
        return TabularStats(num_instances, num_features, (num_features,))
