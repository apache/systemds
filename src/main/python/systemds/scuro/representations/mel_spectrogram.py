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

from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_context_representation_operator,
)


@register_representation(ModalityType.AUDIO)
@register_context_representation_operator(ModalityType.AUDIO)
class MelSpectrogram(UnimodalRepresentation):
    def __init__(self, n_mels=128, hop_length=512, n_fft=2048, params=None):
        parameters = {
            "n_mels": [20, 32, 64, 128],
            "hop_length": [256, 512, 1024, 2048],
            "n_fft": [1024, 2048, 4096],
        }
        super().__init__("MelSpectrogram", ModalityType.TIMESERIES, parameters, False)

        if params is not None:
            n_mels = params.get("n_mels", n_mels)
            hop_length = params.get("hop_length", hop_length)
            n_fft = params.get("n_fft", n_fft)

        self.n_mels = int(n_mels)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        result = []
        metadata_values = list(modality.metadata.values())

        for i, sample in enumerate(modality.data):
            sr = metadata_values[i]["frequency"]
            computed_feature = self.compute_feature(sample, sr)
            result.append(computed_feature)

        transformed_modality.data = result
        return transformed_modality

    def compute_feature(self, instance, sr=None):
        if sr is None:
            sr = 22050
        S = librosa.feature.melspectrogram(
            y=np.asarray(instance),
            sr=sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )
        return S.T

    def get_output_stats(self, input_stats) -> RepresentationStats:
        num_instances = getattr(input_stats, "num_instances", 0)

        if hasattr(input_stats, "max_length"):
            signal_length = input_stats.max_length
        elif hasattr(input_stats, "output_shape") and input_stats.output_shape:
            signal_length = input_stats.output_shape[0]
        else:
            signal_length = 0

        if signal_length <= 0:
            num_frames = 1
        else:
            if signal_length < self.n_fft:
                num_frames = 1
            else:
                num_frames = 1 + (signal_length - self.n_fft) // self.hop_length
            num_frames = max(int(num_frames), 1)

        return RepresentationStats(num_instances, (num_frames, self.n_mels))
