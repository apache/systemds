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
from faster_whisper import WhisperModel
import numpy as np

from systemds.scuro.dataloader.base_loader import BaseLoader
from systemds.scuro.modality.type import ModalityType


class TranscriptLoader(BaseLoader):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        data_type: Union[np.dtype, str] = np.float32,
        chunk_size: Optional[int] = None,
        normalize: bool = True,
        transcribe_model_size: str = "medium",
        load=True,
    ):
        super().__init__(source_path, indices, data_type, chunk_size, ModalityType.TEXT)
        self.model = WhisperModel(
            transcribe_model_size, device="cpu", compute_type="int8"
        )
        self.normalize = normalize
        self.load_data_from_file = load

    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        self.file_sanity_check(file)
        segments, _ = self.model.transcribe(file, vad_filter=True)

        for i, seg in enumerate(segments):
            md = self.modality_type.create_metadata(len(seg.text.split()), seg.text)
            md["timestamp_start"] = seg.start
            md["timestamp_end"] = seg.end
            md["text"] = seg.text

            self.metadata.append(md)

            self.data.append(seg.text)
