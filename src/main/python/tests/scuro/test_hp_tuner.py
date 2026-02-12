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

from systemds.scuro.drsearch.multimodal_optimizer import MultimodalOptimizer
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.unimodal_optimizer import UnimodalOptimizer

from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.covarep_audio_features import (
    ZeroCrossing,
    Spectral,
    Pitch,
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
from systemds.scuro.drsearch.hyperparameter_tuner import HyperparameterTuner

from unittest.mock import patch


class TestHPTuner(unittest.TestCase):
    data_generator = None
    num_instances = 0

    @classmethod
    def setUpClass(cls):
        cls.num_instances = 10
        cls.mods = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]
        cls.indices = np.array(range(cls.num_instances))
        cls.tasks = [
            TestTask("UnimodalRepresentationTask1", "TestSVM1", cls.num_instances),
            TestTask("UnimodalRepresentationTask2", "TestSVM2", cls.num_instances),
        ]

    def test_hp_tuner_for_audio_modality(self):
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, 3000
        )
        audio = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )

        self.run_hp_for_modality([audio])

    def test_multimodal_hp_tuning(self):
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(
            self.num_instances, 3000
        )
        audio = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )

        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )

        # self.run_hp_for_modality(
        #     [audio, text], multimodal=True, tune_unimodal_representations=True
        # )
        self.run_hp_for_modality(
            [audio, text], multimodal=True, tune_unimodal_representations=False
        )

    def test_hp_tuner_for_text_modality(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        self.run_hp_for_modality([text])

    def run_hp_for_modality(
        self, modalities, multimodal=False, tune_unimodal_representations=False
    ):
        with patch.object(
            Registry,
            "_representations",
            {
                ModalityType.TEXT: [W2V, BoW],
                ModalityType.AUDIO: [Spectrogram, ZeroCrossing, Spectral, Pitch],
                ModalityType.TIMESERIES: [ResNet],
                ModalityType.VIDEO: [ResNet],
                ModalityType.EMBEDDING: [],
            },
        ):
            registry = Registry()
            registry._fusion_operators = [LSTM]
            unimodal_optimizer = UnimodalOptimizer(modalities, self.tasks, False)
            unimodal_optimizer.optimize()

            hp = HyperparameterTuner(
                modalities, self.tasks, unimodal_optimizer.operator_performance
            )

            if multimodal:
                m_o = MultimodalOptimizer(
                    modalities,
                    unimodal_optimizer.operator_performance,
                    self.tasks,
                    debug=False,
                    min_modalities=2,
                    max_modalities=3,
                )
                fusion_results = m_o.optimize(20)

                hp.tune_multimodal_representations(
                    fusion_results,
                    k=1,
                    optimize_unimodal=tune_unimodal_representations,
                    max_eval_per_rep=10,
                )

            else:
                hp.tune_unimodal_representations(max_eval_per_rep=10)

            assert len(hp.optimization_results.results) == len(self.tasks)
            if multimodal:
                if tune_unimodal_representations:
                    assert (
                        len(
                            hp.optimization_results.results[self.tasks[0].model.name][0]
                        )
                        == 1
                    )
                else:
                    assert (
                        len(
                            hp.optimization_results.results[self.tasks[0].model.name][
                                "mm_results"
                            ]
                        )
                        == 1
                    )
            else:
                assert (
                    len(hp.optimization_results.results[self.tasks[0].model.name]) == 1
                )
                assert (
                    len(hp.optimization_results.results[self.tasks[0].model.name][0])
                    == 2
                )


if __name__ == "__main__":
    unittest.main()
