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
import librosa
import numpy as np

from systemds.scuro.dataloader.base_loader import BaseLoader
from systemds.scuro.modality.type import ModalityType


@dataclass
class AudioStats:
    sampling_rate: int
    max_length: int
    avg_length: float
    num_instances: int
    output_shape_is_known: bool

    @property
    def output_shape(self):
        return (self.max_length,)


class AudioLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        data_type: Union[np.dtype, str] = np.float32,
        chunk_size: Optional[int] = None,
        normalize: bool = True,
        load=True,
    ):
        super().__init__(
            source_path, indices, data_type, chunk_size, ModalityType.AUDIO
        )
        self.normalize = normalize
        self.load_data_from_file = load
        self.stats = self.get_stats(source_path)

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)
        if not self.load_data_from_file:
            import numpy as np

            audio = np.array([0])
            sr = 1000
        else:
            audio, sr = librosa.load(file, dtype=self._data_type)

            if self.normalize:
                audio = librosa.util.normalize(audio)

        self.metadata.append(self.modality_type.create_metadata(sr, audio))

        self.data.append(audio)

    def get_stats(self, source_path: str):
        sampling_rate = 0
        max_length = 0
        avg_length = 0
        num_instances = 0
        for file in os.listdir(source_path):
            file_name = file.split(".")[0]
            if file_name not in self.indices:
                continue
            self.file_sanity_check(source_path + file)
            audio, sr = librosa.load(source_path + file, dtype=self._data_type)
            num_instances += 1
            sampling_rate = max(sampling_rate, sr)
            max_length = max(max_length, audio.shape[0])
            avg_length += audio.shape[0]
        avg_length /= num_instances
        return AudioStats(sampling_rate, max_length, avg_length, num_instances, True)
