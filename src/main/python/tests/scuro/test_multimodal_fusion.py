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

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.average import Average
from systemds.scuro.drsearch.fusion_optimizer import FusionOptimizer
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.models.model import Model
from systemds.scuro.drsearch.task import Task
from systemds.scuro.drsearch.unimodal_representation_optimizer import (
    UnimodalRepresentationOptimizer,
)

from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.resnet import ResNet
from tests.scuro.data_generator import setup_data

from systemds.scuro.dataloader.audio_loader import AudioLoader
from systemds.scuro.dataloader.video_loader import VideoLoader
from systemds.scuro.dataloader.text_loader import TextLoader
from systemds.scuro.modality.type import ModalityType

from unittest.mock import patch


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


class TestCNN(Model):
    def __init__(self):
        super().__init__("TestCNN")

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


class TestMultimodalRepresentationOptimizer(unittest.TestCase):
    pass
#     test_file_path = None
#     data_generator = None
#     num_instances = 0
#
#     @classmethod
#     def setUpClass(cls):
#         cls.test_file_path = "fusion_optimizer_test_data"
#
#         cls.num_instances = 10
#         cls.mods = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]
#
#         cls.data_generator = setup_data(cls.mods, cls.num_instances, cls.test_file_path)
#         split = train_test_split(
#             cls.data_generator.indices,
#             cls.data_generator.labels,
#             test_size=0.2,
#             random_state=42,
#         )
#         cls.train_indizes, cls.val_indizes = [int(i) for i in split[0]], [
#             int(i) for i in split[1]
#         ]
#
#         cls.tasks = [
#             Task(
#                 "UnimodalRepresentationTask1",
#                 TestSVM(),
#                 cls.data_generator.labels,
#                 cls.train_indizes,
#                 cls.val_indizes,
#             ),
#             Task(
#                 "UnimodalRepresentationTask2",
#                 TestCNN(),
#                 cls.data_generator.labels,
#                 cls.train_indizes,
#                 cls.val_indizes,
#             ),
#         ]
#
#     @classmethod
#     def tearDownClass(cls):
#         shutil.rmtree(cls.test_file_path)
#
#     def test_multimodal_fusion(self):
#         task = Task(
#             "UnimodalRepresentationTask1",
#             TestSVM(),
#             self.data_generator.labels,
#             self.train_indizes,
#             self.val_indizes,
#         )
#         audio_data_loader = AudioLoader(
#             self.data_generator.get_modality_path(ModalityType.AUDIO),
#             self.data_generator.indices,
#         )
#         audio = UnimodalModality(audio_data_loader)
#
#         text_data_loader = TextLoader(
#             self.data_generator.get_modality_path(ModalityType.TEXT),
#             self.data_generator.indices,
#         )
#         text = UnimodalModality(text_data_loader)
#
#         video_data_loader = VideoLoader(
#             self.data_generator.get_modality_path(ModalityType.VIDEO),
#             self.data_generator.indices,
#         )
#         video = UnimodalModality(video_data_loader)
#
#         with patch.object(
#             Registry,
#             "_representations",
#             {
#                 ModalityType.TEXT: [W2V],
#                 ModalityType.AUDIO: [Spectrogram],
#                 ModalityType.TIMESERIES: [ResNet],
#                 ModalityType.VIDEO: [ResNet],
#                 ModalityType.EMBEDDING: [],
#             },
#         ):
#             registry = Registry()
#             registry._fusion_operators = [Average, Concatenation]
#             unimodal_optimizer = UnimodalRepresentationOptimizer(
#                 [text, audio, video], [task], max_chain_depth=2
#             )
#             unimodal_optimizer.optimize()
#
#             multimodal_optimizer = FusionOptimizer(
#                 [audio, text, video],
#                 task,
#                 unimodal_optimizer.optimization_results,
#                 unimodal_optimizer.cache,
#                 2,
#                 2,
#                 debug=False,
#             )
#             multimodal_optimizer.optimize()
#
#
# if __name__ == "__main__":
#     unittest.main()
