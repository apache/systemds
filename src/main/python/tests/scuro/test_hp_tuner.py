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
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.models.model import Model
from systemds.scuro.drsearch.task import Task
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
from tests.scuro.data_generator import ModalityRandomDataGenerator, TestDataLoader

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.hyperparameter_tuner import HyperparameterTuner


class TestSVM(Model):
    def __init__(self):
        super().__init__("TestSVM")

    def fit(self, X, y, X_test, y_test):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        self.clf = svm.SVC(C=1, gamma="scale", kernel="rbf", verbose=False)
        self.clf = self.clf.fit(X, np.array(y))
        y_pred = self.clf.predict(X)

        return classification_report(
            y, y_pred, output_dict=True, digits=3, zero_division=1
        )["accuracy"]

    def test(self, test_X: np.ndarray, test_y: np.ndarray):
        if test_X.ndim > 2:
            test_X = test_X.reshape(test_X.shape[0], -1)
        y_pred = self.clf.predict(np.array(test_X))  # noqa

        return classification_report(
            np.array(test_y), y_pred, output_dict=True, digits=3, zero_division=1
        )["accuracy"]


from unittest.mock import patch


class TestHPTuner(unittest.TestCase):
    data_generator = None
    num_instances = 0

    @classmethod
    def setUpClass(cls):
        cls.num_instances = 10
        cls.mods = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]
        cls.labels = ModalityRandomDataGenerator().create_balanced_labels(
            num_instances=cls.num_instances
        )
        cls.indices = np.array(range(cls.num_instances))

        split = train_test_split(
            cls.indices,
            cls.labels,
            test_size=0.2,
            random_state=42,
        )
        cls.train_indizes, cls.val_indizes = [int(i) for i in split[0]], [
            int(i) for i in split[1]
        ]

        cls.tasks = [
            Task(
                "UnimodalRepresentationTask1",
                TestSVM(),
                cls.labels,
                cls.train_indizes,
                cls.val_indizes,
            )
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

        self.run_hp_for_modality(audio)

    def test_hp_tuner_for_text_modality(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        self.run_hp_for_modality(text)

    def run_hp_for_modality(self, modality):
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

            unimodal_optimizer = UnimodalOptimizer([modality], self.tasks, False)
            unimodal_optimizer.optimize()

            hp = HyperparameterTuner(
                [modality], self.tasks, unimodal_optimizer.operator_performance
            )
            hp.tune_unimodal_representations()


if __name__ == "__main__":
    unittest.main()
