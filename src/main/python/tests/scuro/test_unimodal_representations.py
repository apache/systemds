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


class TestUnimodalRepresentations(unittest.TestCase):
    pass


#     test_file_path = None
#     mods = None
#     text = None
#     audio = None
#     video = None
#     data_generator = None
#     num_instances = 0
#
#     @classmethod
#     def setUpClass(cls):
#         cls.test_file_path = "unimodal_test_data"
#
#         cls.num_instances = 4
#         cls.mods = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]
#
#         cls.data_generator = setup_data(cls.mods, cls.num_instances, cls.test_file_path)
#         os.makedirs(f"{cls.test_file_path}/embeddings")
#
#     @classmethod
#     def tearDownClass(cls):
#         print("Cleaning up test data")
#         shutil.rmtree(cls.test_file_path)
#
#     def test_audio_representations(self):
#         audio_representations = [MelSpectrogram()]  # TODO: add FFT, TFN, 1DCNN
#         audio_data_loader = AudioLoader(
#             self.data_generator.get_modality_path(ModalityType.AUDIO),
#             self.data_generator.indices,
#         )
#         audio = UnimodalModality(audio_data_loader)
#
#         for representation in audio_representations:
#             r = audio.apply_representation(representation)
#             assert r.data is not None
#             assert len(r.data) == self.num_instances
#
#     def test_video_representations(self):
#         video_representations = [ResNet()]  # Todo: add other video representations
#         video_data_loader = VideoLoader(
#             self.data_generator.get_modality_path(ModalityType.VIDEO),
#             self.data_generator.indices,
#         )
#         video = UnimodalModality(video_data_loader)
#         for representation in video_representations:
#             r = video.apply_representation(representation)
#             assert r.data is not None
#             assert len(r.data) == self.num_instances
#
#     def test_text_representations(self):
#         test_representations = [BoW(2, 2), W2V(5, 2, 2), TfIdf(2), Bert()]
#         text_data_loader = TextLoader(
#             self.data_generator.get_modality_path(ModalityType.TEXT),
#             self.data_generator.indices,
#         )
#         text = UnimodalModality(text_data_loader)
#
#         for representation in test_representations:
#             r = text.apply_representation(representation)
#             assert r.data is not None
#             assert len(r.data) == self.num_instances
#
#     def test_chunked_video_representations(self):
#         video_representations = [ResNet()]
#         video_data_loader = VideoLoader(
#             self.data_generator.get_modality_path(ModalityType.VIDEO),
#             self.data_generator.indices,
#             chunk_size=2,
#         )
#         video = UnimodalModality(video_data_loader)
#         for representation in video_representations:
#             r = video.apply_representation(representation)
#             assert r.data is not None
#             assert len(r.data) == self.num_instances
#             assert len(r.metadata) == self.num_instances
#
#
# if __name__ == "__main__":
#     unittest.main()
