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
import librosa
import numpy as np

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.transformed import TransformedModality

from systemds.scuro.representations.unimodal import UnimodalRepresentation


class MelSpectrogram(UnimodalRepresentation):
    def __init__(self):
        super().__init__("MelSpectrogram")

    def transform(self, modality):
        transformed_modality = TransformedModality(
            ModalityType.TIMESERIES, self, modality.metadata
        )
        result = []
        max_length = 0
        for sample in modality.data:
            S = librosa.feature.melspectrogram(y=sample, sr=22050)
            S_dB = librosa.power_to_db(S, ref=np.max)
            if S_dB.shape[-1] > max_length:
                max_length = S_dB.shape[-1]
            result.append(S_dB.T)

        transformed_modality.data = result
        return transformed_modality
