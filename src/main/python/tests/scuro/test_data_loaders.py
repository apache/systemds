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

import os
import shutil
import unittest
import numpy as np

from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.resnet import ResNet
from tests.scuro.data_generator import setup_data

from systemds.scuro.dataloader.audio_loader import AudioLoader
from systemds.scuro.dataloader.video_loader import VideoLoader
from systemds.scuro.dataloader.text_loader import TextLoader
from systemds.scuro.modality.type import ModalityType


class TestDataLoaders(unittest.TestCase):
    test_file_path = None
    mods = None
    text = None
    audio = None
    video = None
    data_generator = None
    num_instances = 0

    @classmethod
    def setUpClass(cls):
        cls.test_file_path = "test_data"
        cls.num_instances = 2
        cls.mods = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]
        cls.data_generator = setup_data(cls.mods, cls.num_instances, cls.test_file_path)

        os.makedirs(f"{cls.test_file_path}/embeddings")

        cls.text_ref = cls.data_generator.modalities_by_type[
            ModalityType.TEXT
        ].apply_representation(Bert())
        cls.audio_ref = cls.data_generator.modalities_by_type[
            ModalityType.AUDIO
        ].apply_representation(MelSpectrogram())
        cls.video_ref = cls.data_generator.modalities_by_type[
            ModalityType.VIDEO
        ].apply_representation(ResNet())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_file_path)

    def test_load_audio_data_from_file(self):
        audio_data_loader = AudioLoader(
            self.data_generator.get_modality_path(ModalityType.AUDIO),
            self.data_generator.indices,
        )
        audio = UnimodalModality(audio_data_loader).apply_representation(
            MelSpectrogram()
        )

        for i in range(0, self.num_instances):
            np.testing.assert_almost_equal(self.audio_ref.data[i], audio.data[i])

    def test_load_video_data_from_file(self):
        video_data_loader = VideoLoader(
            self.data_generator.get_modality_path(ModalityType.VIDEO),
            self.data_generator.indices,
        )
        video = UnimodalModality(video_data_loader).apply_representation(ResNet())

        for i in range(0, self.num_instances):
            np.testing.assert_almost_equal(self.video_ref.data[i], video.data[i])

    def test_load_text_data_from_file(self):
        text_data_loader = TextLoader(
            self.data_generator.get_modality_path(ModalityType.TEXT),
            self.data_generator.indices,
        )
        text = UnimodalModality(text_data_loader).apply_representation(Bert())

        for i in range(0, self.num_instances):
            np.testing.assert_almost_equal(self.text_ref.data[i], text.data[i])


if __name__ == "__main__":
    unittest.main()
