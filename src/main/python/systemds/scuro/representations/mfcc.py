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
from systemds.scuro.drsearch.operator_registry import register_representation


@register_representation(ModalityType.AUDIO)
class MFCC(UnimodalRepresentation):
    def __init__(self, n_mfcc=12, dct_type=2, n_mels=128, hop_length=512):
        parameters = {
            "n_mfcc": [x for x in range(10, 26)],
            "dct_type": [1, 2, 3],
            "hop_length": [256, 512, 1024, 2048],
            "n_mels": [20, 32, 64, 128],
        }  # TODO
        super().__init__("MFCC", ModalityType.TIMESERIES, parameters)
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.n_mels = n_mels
        self.hop_length = hop_length

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        result = []

        for i, sample in enumerate(modality.data):
            sr = list(modality.metadata.values())[i]["frequency"]
            mfcc = librosa.feature.mfcc(
                y=np.array(sample),
                sr=sr,
                n_mfcc=self.n_mfcc,
                dct_type=self.dct_type,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            ).astype(modality.data_type)
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

            result.append(mfcc.T)

        transformed_modality.data = result
        return transformed_modality
