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

from systemds.scuro.representations.mfcc import MFCC
from systemds.scuro.representations.wav2vec import Wav2Vec
from systemds.scuro.representations.window import WindowAggregation
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.tfidf import TfIdf
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.representations.max import RowMax
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.multiplication import Multiplication
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.sum import Sum


class TestOperatorRegistry(unittest.TestCase):
    def test_audio_representations_in_registry(self):
        registry = Registry()
        for representation in [Spectrogram, MelSpectrogram, Wav2Vec, MFCC]:
            assert representation in registry.get_representations(
                ModalityType.AUDIO
            ), f"{representation} not in registry"

    def test_video_representations_in_registry(self):
        registry = Registry()
        assert registry.get_representations(ModalityType.VIDEO) == [ResNet]

    def test_timeseries_representations_in_registry(self):
        registry = Registry()
        assert registry.get_representations(ModalityType.TIMESERIES) == [ResNet]

    def test_text_representations_in_registry(self):
        registry = Registry()
        for representation in [BoW, TfIdf, W2V, Bert]:
            assert representation in registry.get_representations(
                ModalityType.TEXT
            ), f"{representation} not in registry"

    def test_context_operator_in_registry(self):
        registry = Registry()
        assert registry.get_context_operators() == [WindowAggregation]

    # def test_fusion_operator_in_registry(self):
    #     registry = Registry()
    #     for fusion_operator in [
    #         # RowMax,
    #         Sum,
    #         Average,
    #         Concatenation,
    #         LSTM,
    #         Multiplication,
    #     ]:
    #         assert (
    #             fusion_operator in registry.get_fusion_operators()
    #         ), f"{fusion_operator} not in registry"


if __name__ == "__main__":
    unittest.main()
