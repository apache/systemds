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
from systemds.scuro.modality.audio_modality import AudioModality
from systemds.scuro.modality.text_modality import TextModality
from systemds.scuro.modality.video_modality import VideoModality
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.representation_dataloader import HDF5, NPY, Pickle
from tests.scuro.data_generator import TestDataGenerator


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
        cls.video = VideoModality(
            "", ResNet(f"{cls.test_file_path}/embeddings/resnet_embeddings.hdf5")
        )
        cls.audio = AudioModality(
            "",
            MelSpectrogram(
                output_file=f"{cls.test_file_path}/embeddings/mel_sp_embeddings.npy"
            ),
        )
        cls.text = TextModality(
            "",
            Bert(
                avg_layers=4,
                output_file=f"{cls.test_file_path}/embeddings/bert_embeddings.pkl",
            ),
        )
        cls.mods = [cls.video, cls.audio, cls.text]
        cls.data_generator = TestDataGenerator(cls.mods, cls.test_file_path)
        cls.data_generator.create_multimodal_data(cls.num_instances)
        cls.text.read_all(cls.indizes)
        cls.audio.read_all(cls.indizes)
        cls.video.read_all([i for i in range(0, cls.num_instances)])

    @classmethod
    def tearDownClass(cls):
        print("Cleaning up test data")
        shutil.rmtree(cls.test_file_path)

    def test_load_audio_data_from_file(self):
        load_audio = AudioModality(
            f"{self.test_file_path}/embeddings/mel_sp_embeddings.npy", NPY()
        )
        load_audio.read_all(self.indizes)

        for i in range(0, self.num_instances):
            assert round(sum(self.audio.data[i]), 4) == round(
                sum(load_audio.data[i]), 4
            )

    def test_load_video_data_from_file(self):
        load_video = VideoModality(
            f"{self.test_file_path}/embeddings/resnet_embeddings.hdf5", HDF5()
        )
        load_video.read_all(self.indizes)

        for i in range(0, self.num_instances):
            assert round(sum(self.video.data[i]), 4) == round(
                sum(load_video.data[i]), 4
            )

    def test_load_text_data_from_file(self):
        load_text = TextModality(
            f"{self.test_file_path}/embeddings/bert_embeddings.pkl", Pickle()
        )
        load_text.read_all(self.indizes)

        for i in range(0, self.num_instances):
            assert round(sum(self.text.data[i]), 4) == round(sum(load_text.data[i]), 4)


if __name__ == "__main__":
    unittest.main()
