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

from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.drsearch.operator_registry import register_representation

import warnings

warnings.filterwarnings("ignore", message="Some weights of")


@register_representation(ModalityType.AUDIO)
class Wav2Vec(UnimodalRepresentation):
    def __init__(self):
        super().__init__("Wav2Vec", ModalityType.TIMESERIES, {})
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        ).float()

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )

        result = []
        for i, sample in enumerate(modality.data):
            sr = list(modality.metadata.values())[i]["frequency"]
            audio_resampled = librosa.resample(sample, orig_sr=sr, target_sr=16000)
            input = self.processor(
                audio_resampled, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input.input_values = input.input_values.float()
            input.data["input_values"] = input.data["input_values"].float()
            with torch.no_grad():
                outputs = self.model(**input)
                features = outputs.extract_features
                # TODO: check how to get intermediate representations
            result.append(torch.flatten(features.mean(dim=1), 1).detach().cpu().numpy())

        transformed_modality.data = result
        return transformed_modality
