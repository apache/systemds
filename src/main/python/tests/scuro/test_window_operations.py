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

import shutil
import unittest
import math

import numpy as np

from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.tfidf import TfIdf
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.resnet import ResNet
from tests.scuro.data_generator import setup_data

from systemds.scuro.dataloader.audio_loader import AudioLoader
from systemds.scuro.dataloader.video_loader import VideoLoader
from systemds.scuro.dataloader.text_loader import TextLoader
from systemds.scuro.modality.type import ModalityType


class TestWindowOperations(unittest.TestCase):
    test_file_path = None
    mods = None
    text = None
    audio = None
    video = None
    data_generator = None
    num_instances = 0

    @classmethod
    def setUpClass(cls):
        cls.test_file_path = "window_operation_test_data"

        cls.num_instances = 4
        cls.mods = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]

        cls.data_generator = setup_data(cls.mods, cls.num_instances, cls.test_file_path)
        cls.aggregations = ["mean", "sum", "max", "min"]

    @classmethod
    def tearDownClass(cls):
        print("Cleaning up test data")
        shutil.rmtree(cls.test_file_path)

    def test_window_operations_on_audio_representations(self):
        window_size = 10
        audio_representations = [MelSpectrogram()]
        audio_data_loader = AudioLoader(
            self.data_generator.get_modality_path(ModalityType.AUDIO),
            self.data_generator.indices,
        )
        audio = UnimodalModality(audio_data_loader)

        for representation in audio_representations:
            self.run_window_operations_for_modality(audio, representation, window_size)

    def test_window_operations_on_video_representations(self):
        window_size = 10
        video_representations = [ResNet()]
        video_data_loader = VideoLoader(
            self.data_generator.get_modality_path(ModalityType.VIDEO),
            self.data_generator.indices,
        )
        video = UnimodalModality(video_data_loader)

        for representation in video_representations:
            self.run_window_operations_for_modality(video, representation, window_size)

    def test_window_operations_on_text_representations(self):
        window_size = 10
        text_representations = [BoW(2, 2), TfIdf(2, 2), Bert(), W2V(2, 1, 2)]
        text_data_loader = TextLoader(
            self.data_generator.get_modality_path(ModalityType.TEXT),
            self.data_generator.indices,
        )
        text = UnimodalModality(text_data_loader)

        for representation in text_representations:
            self.run_window_operations_for_modality(text, representation, window_size)

    def run_window_operations_for_modality(self, modality, representation, window_size):
        r = modality.apply_representation(representation)
        for aggregation in self.aggregations:
            windowed_modality = r.window(window_size, aggregation)

            self.verify_window_operation(aggregation, r, windowed_modality, window_size)

    def verify_window_operation(
        self, aggregation, modality, windowed_modality, window_size
    ):
        assert windowed_modality.data is not None
        assert len(windowed_modality.data) == self.num_instances

        for i, instance in enumerate(windowed_modality.data):
            assert (
                list(windowed_modality.metadata.values())[i]["data_layout"]["shape"][1]
                == list(modality.metadata.values())[i]["data_layout"]["shape"][1]
            )
            assert len(instance) == math.ceil(len(modality.data[i]) / window_size)
            for j in range(0, len(instance)):
                if aggregation == "mean":
                    np.testing.assert_almost_equal(
                        instance[j],
                        np.mean(
                            modality.data[i][j * window_size : (j + 1) * window_size],
                            axis=0,
                        ),
                    )
                elif aggregation == "sum":
                    np.testing.assert_almost_equal(
                        instance[j],
                        np.sum(
                            modality.data[i][j * window_size : (j + 1) * window_size],
                            axis=0,
                        ),
                    )
                elif aggregation == "max":
                    np.testing.assert_almost_equal(
                        instance[j],
                        np.max(
                            modality.data[i][j * window_size : (j + 1) * window_size],
                            axis=0,
                        ),
                    )
                elif aggregation == "min":
                    np.testing.assert_almost_equal(
                        instance[j],
                        np.min(
                            modality.data[i][j * window_size : (j + 1) * window_size],
                            axis=0,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
