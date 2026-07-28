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
from systemds.scuro.utils.static_variables import (
    NP_ARRAY_HEADER_BYTES,
    PY_LIST_HEADER_BYTES,
    PY_LIST_SLOT_BYTES,
)


@register_representation(ModalityType.AUDIO)
@register_context_representation_operator(ModalityType.AUDIO)
class MFCC(UnimodalRepresentation):
    def __init__(self, n_mfcc=12, dct_type=2, n_mels=128, hop_length=512, params=None):
        parameters = {
            "n_mfcc": [x for x in range(10, 26)],
            "dct_type": [1, 2, 3],
            "hop_length": [256, 512, 1024, 2048],
            "n_mels": [20, 32, 64, 128],
        }  # TODO
        super().__init__("MFCC", ModalityType.TIMESERIES, parameters, False)

        # Allow construction from a parameter dict (used by optimizer)
        if params is not None:
            n_mfcc = params.get("n_mfcc", n_mfcc)
            dct_type = params.get("dct_type", dct_type)
            n_mels = params.get("n_mels", n_mels)
            hop_length = params.get("hop_length", hop_length)

        self.n_mfcc = int(n_mfcc)
        self.dct_type = int(dct_type)
        self.n_mels = int(n_mels)
        self.hop_length = int(hop_length)

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        result = []

        for i, sample in enumerate(modality.data):
            sr = modality.metadata[i]["frequency"]
            computed_feature = self.compute_feature(sample, sr)
            result.append(computed_feature)

        transformed_modality.data = result
        return transformed_modality

    def compute_feature(self, instance, sr=None):
        if sr is None:
            sr = 22050
        mfcc = librosa.feature.mfcc(
            y=np.array(instance),
            sr=sr,
            n_mfcc=self.n_mfcc,
            dct_type=self.dct_type,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        if mfcc.ndim == 2:
            mean = np.mean(mfcc, keepdims=True)
            std = np.std(mfcc, keepdims=True)
        else:
            mean = np.mean(mfcc, axis=(1, 2), keepdims=True)
            std = np.std(mfcc, axis=(1, 2), keepdims=True)
        mfcc = (mfcc - mean) / np.maximum(std, 1e-8)

        if instance.ndim == 1:
            return mfcc.T

        return mfcc.transpose(0, 2, 1)

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
            num_frames = 1 + max(int((signal_length - 1) // self.hop_length), 0)
            num_frames = max(int(num_frames), 1)

        return RepresentationStats(num_instances, (num_frames, self.n_mfcc))

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        n = int(getattr(input_stats, "num_instances", 0))
        if hasattr(input_stats, "max_length"):
            signal_length = int(getattr(input_stats, "max_length", 0))
        elif hasattr(input_stats, "output_shape") and input_stats.output_shape:
            signal_length = int(input_stats.output_shape[0])
        else:
            signal_length = 0

        if signal_length <= 0:
            num_frames = 1
        else:
            num_frames = 1 + max(int((signal_length - 1) // self.hop_length), 0)
            num_frames = max(int(num_frames), 1)

        out_elem = np.dtype(np.float32).itemsize
        n_fft = 2048
        num_freq_bins = 1 + n_fft // 2
        output_payload_per_instance = num_frames * self.n_mfcc * out_elem
        retained_output_bytes = PY_LIST_HEADER_BYTES + n * (
            output_payload_per_instance + NP_ARRAY_HEADER_BYTES + PY_LIST_SLOT_BYTES
        )

        input_copy_bytes = max(signal_length, 1) * out_elem
        stft_bytes = num_frames * num_freq_bins * np.dtype(np.complex64).itemsize
        magnitude_bytes = num_frames * num_freq_bins * out_elem
        mel_projection_bytes = num_frames * self.n_mels * out_elem
        dct_output_bytes = output_payload_per_instance
        norm_workspace_bytes = max(dct_output_bytes, 256 * 1024)
        fft_workspace_bytes = max(2 * 1024 * 1024, stft_bytes // 2)

        transient_one_instance = (
            input_copy_bytes
            + stft_bytes
            + magnitude_bytes
            + mel_projection_bytes
            + dct_output_bytes
            + norm_workspace_bytes
            + fft_workspace_bytes
        )
        cpu_peak = int(
            (retained_output_bytes + transient_one_instance) * 1.15 + 16 * 1024 * 1024
        )
        return {
            "cpu_peak_bytes": cpu_peak,
            "gpu_peak_bytes": 0,
        }
