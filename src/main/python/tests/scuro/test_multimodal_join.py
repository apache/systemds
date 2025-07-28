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

# TODO: Test edge cases: unequal number of audio-video timestamps (should still work and add the average over all audio/video samples)

import shutil
import unittest

import numpy as np
import copy
from systemds.scuro.modality.joined import JoinCondition
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.resnet import ResNet
from tests.scuro.data_generator import TestDataLoader, ModalityRandomDataGenerator

from systemds.scuro.dataloader.audio_loader import AudioLoader
from systemds.scuro.dataloader.video_loader import VideoLoader
from systemds.scuro.modality.type import ModalityType


class TestMultimodalJoin(unittest.TestCase):
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
        cls.num_instances = 4
        cls.indices = np.array(range(cls.num_instances))
        cls.audio_data, cls.audio_md = ModalityRandomDataGenerator().create_audio_data(
            cls.num_instances, 32000
        )
        
        cls.video_data, cls.video_md = ModalityRandomDataGenerator().create_visual_modality(
            cls.num_instances, 60
        )

    def test_video_audio_join(self):
        self._execute_va_join()

    def test_chunked_video_audio_join(self):
        self._execute_va_join(2)

    def test_video_chunked_audio_join(self):
        self._execute_va_join(None, 2)

    def test_chunked_video_chunked_audio_join(self):
        self._execute_va_join(2, 2)

    def test_audio_video_join(self):
        # Audio has a much higher frequency than video, hence we would need to
        # duplicate or interpolate frames to match them to the audio frequency
        self._execute_av_join()

    # TODO
    # def test_chunked_audio_video_join(self):
    #     self._execute_av_join(2)

    # TODO
    # def test_chunked_audio_chunked_video_join(self):
    #     self._execute_av_join(2, 2)

    def _execute_va_join(self, l_chunk_size=None, r_chunk_size=None):
        video, audio = self._prepare_data(l_chunk_size, r_chunk_size)
        self._join(video, audio, 2)

    def _execute_av_join(self, l_chunk_size=None, r_chunk_size=None):
        video, audio = self._prepare_data(l_chunk_size, r_chunk_size)
        self._join(audio, video, 2)

    def _prepare_data(self, l_chunk_size=None, r_chunk_size=None):
        audio = UnimodalModality(
            TestDataLoader(
                self.indices, r_chunk_size, ModalityType.AUDIO, copy.deepcopy(self.audio_data), np.float32, copy.deepcopy(self.audio_md)
            )
        )
        video = UnimodalModality(
            TestDataLoader(
                self.indices, l_chunk_size, ModalityType.VIDEO, copy.deepcopy(self.video_data), np.float32, copy.deepcopy(self.video_md)
            )
        )

        mel_audio = audio.apply_representation(MelSpectrogram())

        return video, mel_audio

    def _join(self, left_modality, right_modality, window_size):
        resnet_modality = (
            left_modality.join(
                right_modality, JoinCondition("timestamp", "timestamp", "<")
            )
            .apply_representation(ResNet(layer="layer1.0.conv2", model_name="ResNet18"))
            .window_aggregation(window_size, "mean")
            .combine("concat")
        )

        assert resnet_modality.left_modality is not None
        assert resnet_modality.right_modality is not None
        assert len(resnet_modality.left_modality.data) == self.num_instances
        assert len(resnet_modality.right_modality.data) == self.num_instances
        assert resnet_modality.data is not None

        return resnet_modality


if __name__ == "__main__":
    unittest.main()
