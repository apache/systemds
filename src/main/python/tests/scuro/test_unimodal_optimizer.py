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
from systemds.scuro.representations.timeseries_representations import (
    Mean,
    ACF,
)
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.unimodal_optimizer import UnimodalOptimizer

from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.covarep_audio_features import (
    ZeroCrossing,
)
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.bow import BoW
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.resnet import ResNet
from tests.scuro.data_generator import (
    ModalityRandomDataGenerator,
    TestDataLoader,
    TestTask,
)

from systemds.scuro.modality.type import ModalityType

from unittest.mock import patch


class TestUnimodalRepresentationOptimizer(unittest.TestCase):
    data_generator = None
    num_instances = 0

    @classmethod
    def setUpClass(cls):
        cls.num_instances = 10
        cls.mods = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]

        cls.indices = np.array(range(cls.num_instances))

        cls.tasks = [
            TestTask("UnimodalRepresentationTask1", "Test1", cls.num_instances),
            TestTask("UnimodalRepresentationTask2", "Test2", cls.num_instances),
        ]

    def test_unimodal_optimizer_for_audio_modality(self):
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, 3000
        )
        audio = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )

        self.optimize_unimodal_representation_for_modality(audio)

    def test_unimodal_optimizer_for_text_modality(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        self.optimize_unimodal_representation_for_modality(text)

    def test_unimodal_optimizer_for_ts_modality(self):
        ts_data, ts_md = ModalityRandomDataGenerator().create_timeseries_data(
            self.num_instances, 1000
        )
        ts = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TIMESERIES, ts_data, np.float32, ts_md
            )
        )
        self.optimize_unimodal_representation_for_modality(ts)

    def test_unimodal_optimizer_for_video_modality(self):
        video_data, video_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 10, 10
        )
        video = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.VIDEO, video_data, np.float32, video_md
            )
        )
        self.optimize_unimodal_representation_for_modality(video)

    def optimize_unimodal_representation_for_modality(self, modality):
        with patch.object(
            Registry,
            "_representations",
            {
                ModalityType.TEXT: [W2V, BoW],
                ModalityType.AUDIO: [Spectrogram, ZeroCrossing],
                ModalityType.TIMESERIES: [Mean, ACF],
                ModalityType.VIDEO: [ResNet],
                ModalityType.EMBEDDING: [],
            },
        ):
            registry = Registry()

            unimodal_optimizer = UnimodalOptimizer([modality], self.tasks, False)
            unimodal_optimizer.optimize()

            assert (
                unimodal_optimizer.operator_performance.modality_ids[0]
                == modality.modality_id
            )
            assert len(unimodal_optimizer.operator_performance.task_names) == 2
            result, cached = unimodal_optimizer.operator_performance.get_k_best_results(
                modality, 1, self.tasks[0], "accuracy"
            )
            assert len(result) == 1
            assert len(cached) == 1

    # Todo: Add a test with all representations at once
    # Todo: Add test with only one model


if __name__ == "__main__":
    unittest.main()
