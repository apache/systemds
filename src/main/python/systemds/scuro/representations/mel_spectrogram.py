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
import numpy as np

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.transformed import TransformedModality

from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.spectral import mel_spectrogram
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_context_representation_operator,
)
from systemds.scuro.drsearch.process_cache import data_fingerprint
from systemds.scuro.utils.static_variables import (
    NP_ARRAY_HEADER_BYTES,
    PY_LIST_HEADER_BYTES,
    PY_LIST_SLOT_BYTES,
)

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"n_fft=\d+ is too (?:small|large) for input signal",
    module=r"librosa",
)


def power_mel_spectrogram(instance, sr, n_mels, hop_length, n_fft):
    """Power mel spectrogram of one signal, shaped (..., n_mels, frames)."""
    S = mel_spectrogram(
        np.asarray(instance, dtype=np.float32),
        22050 if sr is None else sr,
        n_fft,
        hop_length,
        n_mels,
    )
    return np.swapaxes(S, -2, -1)


def mel_spectrograms(modality, n_mels, hop_length, n_fft, cache=None):
    """Power mel spectrogram of every instance of ``modality``.

    MFCC and MelSpectrogram configurations with equal (n_fft, hop_length,
    n_mels) need the same spectrograms, and they are most of an MFCC's cost.
    With ``cache`` (a BoundedProcessCache) they are computed once per data
    chunk and reused by every such configuration. Callers must not modify
    the returned arrays.
    """
    samples = modality.data
    srs = [
        modality.metadata[i]["frequency"] if modality.metadata else None
        for i in range(len(samples))
    ]

    def compute():
        return [
            power_mel_spectrogram(sample, sr, n_mels, hop_length, n_fft)
            for sample, sr in zip(samples, srs)
        ]

    if cache is None:
        return compute()
    key = (
        "mel_spectrogram",
        int(n_fft),
        int(hop_length),
        int(n_mels),
        data_fingerprint(samples, srs),
    )
    return cache.get_or_compute(key, compute)


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
        self.window_size = self.n_fft

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        spectrograms = mel_spectrograms(
            modality,
            self.n_mels,
            self.hop_length,
            self.n_fft,
            self.intermediate_cache,
        )
        result = [np.swapaxes(S, -2, -1) for S in spectrograms]
        if self.intermediate_cache is not None:
            # Do not hand out views of cached spectrograms.
            result = [S.copy() for S in result]
        transformed_modality.data = result
        return transformed_modality

    def compute_feature(self, instance, sr=None):
        S = power_mel_spectrogram(
            instance, sr, self.n_mels, self.hop_length, self.n_fft
        )
        return np.swapaxes(S, -2, -1)

    def compute_features_batched(self, data, sr=None):
        return self.compute_feature(np.asarray(data), sr=sr)

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
                num_frames = 1 + signal_length // self.hop_length
            num_frames = max(int(num_frames), 1)

        return RepresentationStats(num_instances, (num_frames, self.n_mels))

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
        elif signal_length < self.n_fft:
            num_frames = 1
        else:
            num_frames = 1 + (signal_length) // self.hop_length
            num_frames = max(int(num_frames), 1)

        out_elem = np.dtype(np.float32).itemsize
        num_freq_bins = 1 + self.n_fft // 2
        output_payload_per_instance = num_frames * self.n_mels * out_elem
        retained_output_bytes = PY_LIST_HEADER_BYTES + n * (
            output_payload_per_instance + NP_ARRAY_HEADER_BYTES + PY_LIST_SLOT_BYTES
        )

        input_copy_bytes = max(signal_length, 1) * out_elem
        stft_bytes = num_frames * num_freq_bins * np.dtype(np.complex64).itemsize
        power_spec_bytes = num_frames * num_freq_bins * out_elem
        mel_output_bytes = output_payload_per_instance
        fft_workspace_bytes = max(2 * 1024 * 1024, stft_bytes // 2)

        transient_one_instance = (
            input_copy_bytes
            + stft_bytes
            + power_spec_bytes
            + mel_output_bytes
            + fft_workspace_bytes
        )
        cpu_peak = int(
            (retained_output_bytes + transient_one_instance) * 2 + 12 * 1024 * 1024
        )
        return {
            "cpu_peak_bytes": cpu_peak,
            "gpu_peak_bytes": 0,
        }
