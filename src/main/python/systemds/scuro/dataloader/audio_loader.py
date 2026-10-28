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
from typing import List, Optional, Union
import librosa
import numpy as np

from systemds.scuro.dataloader.base_loader import BaseLoader, LazyFileSequence
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
        sampling_rate: Optional[int] = None,
    ):
        super().__init__(
            source_path, indices, data_type, chunk_size, ModalityType.AUDIO
        )
        self.normalize = normalize
        self.load_data_from_file = load
        self.sampling_rate = sampling_rate
        files = self.get_file_names(self.indices)
        self.stats = self.cached_stats(
            files,
            {
                "dtype": str(data_type),
                "normalize": normalize,
                "sampling_rate": sampling_rate,
            },
            lambda: self.get_stats(source_path),
        )

    def _load(self, indices: List[str]):
        if self.data is not None and len(self.data) == len(indices):
            return self.data, self.metadata

        file_names = self.get_file_names(indices)
        if isinstance(file_names, str):
            file_names = [file_names]
        decoded = LazyFileSequence(file_names, self._decode_file).materialize()
        self.data = [audio for audio, _ in decoded]
        self.metadata = [metadata for _, metadata in decoded]
        return self.data, self.metadata

    def _decode_file(self, file: str):
        self.file_sanity_check(file)
        if not self.load_data_from_file:
            audio = np.array([0])
            sr = 1000
        else:
            audio, sr = librosa.load(file, sr=self.sampling_rate, dtype=self._data_type)

            if self.normalize:
                audio = librosa.util.normalize(audio)

        return audio, self.modality_type.create_metadata(sr, audio)

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        audio, metadata = self._decode_file(file)
        self.metadata.append(metadata)
        self.data.append(audio)

    def get_stats(self, source_path: str):
        sampling_rate = 0
        max_length = 0
        avg_length = 0
        num_instances = 0
        import soundfile

        for file in self.get_file_names(self.indices):
            self.file_sanity_check(file)
            try:
                info = soundfile.info(file)
                sr = self.sampling_rate or info.samplerate
                length = math.ceil(info.frames * sr / info.samplerate)
            except (RuntimeError, TypeError):
                audio, sr = librosa.load(
                    file, sr=self.sampling_rate, dtype=self._data_type
                )
                length = audio.shape[0]
            num_instances += 1
            sampling_rate = max(sampling_rate, sr)
            max_length = max(max_length, length)
            avg_length += length
        avg_length /= num_instances
        return AudioStats(sampling_rate, max_length, avg_length, num_instances, True)
