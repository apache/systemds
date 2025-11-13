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

import unittest

from systemds.scuro import FrequencyMagnitude
from systemds.scuro.representations.covarep_audio_features import (
    ZeroCrossing,
    Spectral,
    Pitch,
    RMSE,
)
from systemds.scuro.representations.mfcc import MFCC
from systemds.scuro.representations.swin_video_transformer import SwinVideoTransformer
from systemds.scuro.representations.clip import CLIPText, CLIPVisual
from systemds.scuro.representations.vgg import VGG19
from systemds.scuro.representations.x3d import X3D, I3D
from systemds.scuro.representations.wav2vec import Wav2Vec
from systemds.scuro.representations.window_aggregation import (
    WindowAggregation,
    StaticWindow,
    DynamicWindow,
)
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.tfidf import TfIdf
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.representations.timeseries_representations import (
    Max,
    Mean,
    Min,
    RMS,
    Sum,
    Std,
    Skew,
    Kurtosis,
    SpectralCentroid,
    BandpowerFFT,
    ACF,
    Quantile,
    ZeroCrossingRate,
    FrequencyMagnitude,
)
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.representations.max import RowMax
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.hadamard import Hadamard
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.multimodal_attention_fusion import AttentionFusion


class TestOperatorRegistry(unittest.TestCase):
    def test_audio_representations_in_registry(self):
        registry = Registry()
        assert registry.get_representations(ModalityType.AUDIO) == [
            MelSpectrogram,
            MFCC,
            Spectrogram,
            Wav2Vec,
            Spectral,
            ZeroCrossing,
            RMSE,
            Pitch,
        ]

    def test_video_representations_in_registry(self):
        registry = Registry()
        assert registry.get_representations(ModalityType.VIDEO) == [
            ResNet,
            SwinVideoTransformer,
            X3D,
            VGG19,
            CLIPVisual,
        ]

    def test_timeseries_representations_in_registry(self):
        registry = Registry()
        assert registry.get_representations(ModalityType.TIMESERIES) == [
            Mean,
            Min,
            Max,
            Sum,
            Std,
            Skew,
            Quantile,
            Kurtosis,
            RMS,
            ZeroCrossingRate,
            ACF,
            FrequencyMagnitude,
            SpectralCentroid,
            BandpowerFFT,
        ]

    def test_text_representations_in_registry(self):
        registry = Registry()
        for representation in [CLIPText, BoW, TfIdf, W2V, Bert]:
            assert representation in registry.get_representations(
                ModalityType.TEXT
            ), f"{representation} not in registry"

    def test_context_operator_in_registry(self):
        registry = Registry()
        assert registry.get_context_operators() == [
            WindowAggregation,
            StaticWindow,
            DynamicWindow,
        ]

    # def test_fusion_operator_in_registry(self):
    #     registry = Registry()
    #     assert registry.get_fusion_operators() == [
    #         Average,
    #         Concatenation,
    #         LSTM,
    #         RowMax,
    #         Hadamard,
    #         Sum,
    #         AttentionFusion,
    #     ]


if __name__ == "__main__":
    unittest.main()
