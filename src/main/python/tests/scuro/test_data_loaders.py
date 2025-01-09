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
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.resnet import ResNet
from tests.scuro.data_generator import TestDataGenerator

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
    indizes = []

    @classmethod
    def setUpClass(cls):
        cls.test_file_path = "test_data"

        if os.path.isdir(cls.test_file_path):
            shutil.rmtree(cls.test_file_path)

        os.makedirs(f"{cls.test_file_path}/embeddings")

        cls.num_instances = 2
        cls.indizes = [str(i) for i in range(0, cls.num_instances)]

        video_data_loader = VideoLoader(cls.test_file_path + "/VIDEO/", cls.indizes)
        audio_data_loader = AudioLoader(cls.test_file_path + "/AUDIO/", cls.indizes)
        text_data_loader = TextLoader(cls.test_file_path + "/TEXT/", cls.indizes)

        # Load modalities (audio, video, text)
        video = UnimodalModality(video_data_loader, ModalityType.VIDEO)
        audio = UnimodalModality(audio_data_loader, ModalityType.AUDIO)
        text = UnimodalModality(text_data_loader, ModalityType.TEXT)

        cls.mods = [video, audio, text]
        cls.data_generator = TestDataGenerator(cls.mods, cls.test_file_path)
        cls.data_generator.create_multimodal_data(cls.num_instances)
        cls.text_ref = text.apply_representation(Bert())
        cls.audio_ref = audio.apply_representation(MelSpectrogram())
        cls.video_ref = video.apply_representation(ResNet())

    @classmethod
    def tearDownClass(cls):
        print("Cleaning up test data")
        shutil.rmtree(cls.test_file_path)

    def test_load_audio_data_from_file(self):
        audio_data_loader = AudioLoader(self.test_file_path + "/audio/", self.indizes)
        audio = UnimodalModality(
            audio_data_loader, ModalityType.AUDIO
        ).apply_representation(MelSpectrogram())

        for i in range(0, self.num_instances):
            assert round(sum(self.audio_ref.data[i]), 4) == round(sum(audio.data[i]), 4)

    def test_load_video_data_from_file(self):
        video_data_loader = VideoLoader(self.test_file_path + "/video/", self.indizes)
        video = UnimodalModality(
            video_data_loader, ModalityType.VIDEO
        ).apply_representation(ResNet())

        for i in range(0, self.num_instances):
            assert round(sum(self.video_ref.data[i]), 4) == round(sum(video.data[i]), 4)

    def test_load_text_data_from_file(self):
        text_data_loader = TextLoader(self.test_file_path + "/text/", self.indizes)
        text = UnimodalModality(
            text_data_loader, ModalityType.TEXT
        ).apply_representation(Bert())

        for i in range(0, self.num_instances):
            assert round(sum(self.text_ref.data[i]), 4) == round(sum(text.data[i]), 4)


if __name__ == "__main__":
    unittest.main()
