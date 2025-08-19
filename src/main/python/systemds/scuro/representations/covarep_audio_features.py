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
class Spectral(UnimodalRepresentation):
    def __init__(self, hop_length=512):
        parameters = {
            "hop_length": [256, 512, 1024, 2048],
        }  # TODO
        super().__init__("Spectral", ModalityType.EMBEDDING, parameters)
        self.hop_length = hop_length

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        result = []
        for i, y in enumerate(modality.data):
            sr = list(modality.metadata.values())[i]["frequency"]

            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=self.hop_length
            )
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, hop_length=self.hop_length
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, hop_length=self.hop_length
            )
            spectral_flatness = librosa.feature.spectral_flatness(
                y=y, hop_length=self.hop_length
            )

            features = np.vstack(
                [
                    spectral_centroid,
                    spectral_bandwidth,
                    spectral_rolloff,
                    spectral_flatness,
                ]
            )

            result.append(features.T)

        transformed_modality.data = result

        return transformed_modality


@register_representation(ModalityType.AUDIO)
class ZeroCrossing(UnimodalRepresentation):
    def __init__(self, hop_length=512):
        parameters = {
            "hop_length": [256, 512, 1024, 2048],
        }  # TODO
        super().__init__("ZeroCrossing", ModalityType.EMBEDDING, parameters)
        self.hop_length = hop_length

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        result = []
        for i, y in enumerate(modality.data):
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y, hop_length=self.hop_length
            )

            result.append(zero_crossing_rate)

        transformed_modality.data = result

        return transformed_modality


@register_representation(ModalityType.AUDIO)
class RMSE(UnimodalRepresentation):
    def __init__(self, frame_length=1024, hop_length=512):
        parameters = {
            "frame_length": [1024, 2048, 4096],
            "hop_length": [256, 512, 1024, 2048],
        }  # TODO
        super().__init__("RMSE", ModalityType.EMBEDDING, parameters)
        self.hop_length = hop_length
        self.frame_length = frame_length

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        result = []
        for i, y in enumerate(modality.data):
            rmse = librosa.feature.rms(
                y=y, frame_length=self.frame_length, hop_length=self.hop_length
            )
            result.append(rmse)

        transformed_modality.data = result

        return transformed_modality


@register_representation(ModalityType.AUDIO)
class Pitch(UnimodalRepresentation):
    def __init__(self, hop_length=512):
        parameters = {
            "hop_length": [256, 512, 1024, 2048],
        }  # TODO
        super().__init__("Pitch", ModalityType.EMBEDDING, parameters)
        self.hop_length = hop_length

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        result = []
        for i, y in enumerate(modality.data):
            sr = list(modality.metadata.values())[i]["frequency"]

            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr, hop_length=self.hop_length
            )
            pitch = pitches[magnitudes.argmax(axis=0), np.arange(magnitudes.shape[1])]

            result.append(pitch[np.newaxis, :])

        transformed_modality.data = result

        return transformed_modality
