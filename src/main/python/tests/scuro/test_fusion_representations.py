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

import numpy as np
from systemds.scuro.representations.window_aggregation import WindowAggregation
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.wav2vec import Wav2Vec
from systemds.scuro.representations.tfidf import TfIdf
from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.mfcc import MFCC
from systemds.scuro.representations.multimodal_attention_fusion import (
    MultiModalAttentionFusion,
    AttentionFusion,
)
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.modality.type import ModalityType
from tests.scuro.data_generator import ModalityRandomDataGenerator, TestDataLoader


class TestFusionOrders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_instances = 40
        cls.indices = np.array(range(cls.num_instances))
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
            cls.num_instances, 100
        )
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            cls.num_instances
        )
        video_data, video_md = ModalityRandomDataGenerator().create_visual_modality(
            cls.num_instances, 60
        )
        cls.audio = UnimodalModality(
            TestDataLoader(
                cls.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )
        cls.video = UnimodalModality(
            TestDataLoader(
                cls.indices, 10, ModalityType.VIDEO, video_data, np.float32, video_md
            )
        )
        cls.text = UnimodalModality(
            TestDataLoader(
                cls.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )

    def test_attention(self):
        r_a = self.audio.apply_representation(MelSpectrogram())
        r_t = self.text.apply_representation(TfIdf())
        r_v = self.video.apply_representation(ResNet())

        fused = AttentionFusion().transform([r_a, r_v, r_t])
