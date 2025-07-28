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

import librosa
import numpy as np

from systemds.scuro.dataloader.base_loader import BaseLoader
from systemds.scuro.modality.type import ModalityType


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

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)
        # if not self.load_data_from_file:
        #     import numpy as np
        #
        #     self.metadata[file] = self.modality_type.create_audio_metadata(
        #         1000, np.array([0])
        #     )
        # else:
        audio, sr = librosa.load(file, dtype=self._data_type)

        if self.normalize:
            audio = librosa.util.normalize(audio)

        self.metadata[file] = self.modality_type.create_audio_metadata(sr, audio)

        self.data.append(audio)
