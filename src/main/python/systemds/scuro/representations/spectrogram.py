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
class Spectrogram(UnimodalRepresentation):
    def __init__(self, hop_length=512, n_fft=2048, params=None):
        parameters = {"hop_length": [256, 512, 1024, 2048], "n_fft": [1024, 2048, 4096]}
        super().__init__("Spectrogram", ModalityType.TIMESERIES, parameters, False)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        result = []

        for i, sample in enumerate(modality.data):
            computed_feature = self.compute_feature(sample)
            result.append(computed_feature)

        transformed_modality.data = result
        return transformed_modality

    def compute_feature(self, instance):
        spectrogram = librosa.stft(
            y=np.array(np.abs(instance)), hop_length=self.hop_length, n_fft=self.n_fft
        )
        return librosa.amplitude_to_db(np.abs(spectrogram)).T

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

        num_freq_bins = 1 + self.n_fft // 2
        return RepresentationStats(num_instances, (num_frames, num_freq_bins))
