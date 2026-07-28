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
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import torch
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.transformed import TransformedModality

from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.drsearch.operator_registry import register_representation

import warnings

warnings.filterwarnings("ignore", message="Some weights of")


@register_representation(ModalityType.AUDIO)
class Wav2Vec(UnimodalRepresentation):
    def __init__(self, params=None):
        super().__init__("Wav2Vec", ModalityType.TIMESERIES, {})
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        ).float()

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )

        result = []
        for i, sample in enumerate(modality.data):
            sr = modality.metadata[i]["frequency"]
            audio_resampled = librosa.resample(
                np.array(sample), orig_sr=sr, target_sr=16000
            )
            input = self.processor(
                audio_resampled, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input.input_values = input.input_values.float()
            input.data["input_values"] = input.data["input_values"].float()
            with torch.no_grad():
                outputs = self.model(**input)
                features = outputs.extract_features
                # TODO: check how to get intermediate representations
            result.append(torch.flatten(features.mean(dim=1)).detach().cpu().numpy())

        transformed_modality.data = np.array(result)
        return transformed_modality

    def get_output_stats(self, input_stats) -> RepresentationStats:
        num_instances = getattr(input_stats, "num_instances", 0)
        embedding_dim = 512
        return RepresentationStats(num_instances, (embedding_dim,))

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        n = int(getattr(input_stats, "num_instances", 1))

        if hasattr(input_stats, "max_length"):
            signal_len = int(getattr(input_stats, "max_length", 16000))
        elif hasattr(input_stats, "output_shape") and input_stats.output_shape:
            signal_len = int(input_stats.output_shape[0])
        else:
            signal_len = 16000
        signal_len = max(signal_len, 1)

        hidden = 768
        stride = 320  # conv frontend effective stride
        frames = max(1, int(np.ceil(signal_len / stride)))

        model_resident = 420 * 1024 * 1024  # ~420 MB
        activation_bytes = int(frames * hidden * 4 * 24)
        io_temp = int(signal_len * 4 * 4) + 16 * 1024 * 1024

        output_bytes = n * 512 * 4

        cpu_peak = int(
            (model_resident + activation_bytes + io_temp + output_bytes) * 1.25
        )

        gpu_peak = 0
        cpu_peak = max(cpu_peak, 600 * 1024 * 1024)

        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": gpu_peak}
