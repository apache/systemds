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
import copy
import time
from typing import List, Union
from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.representation import Representation
from systemds.scuro.models.model import Model
import numpy as np
from sklearn.model_selection import KFold


class PerformanceMeasure:
    def __init__(self, name, metrics, higher_is_better=True):
        self.average_scores = None
        self.name = name
        self.metrics = metrics
        self.higher_is_better = higher_is_better
        self.scores = {}

        if isinstance(metrics, list):
            for metric in metrics:
                self.scores[metric] = []
        else:
            self.scores[metrics] = []

    def add_scores(self, scores):
        if isinstance(self.metrics, list):
            for metric in self.metrics:
                self.scores[metric].append(scores[metric])
        else:
            self.scores[self.metrics].append(scores[self.metrics])

    def compute_averages(self):
        self.average_scores = {}
        if isinstance(self.metrics, list):
            for metric in self.metrics:
                self.average_scores[metric] = np.mean(self.scores[metric])
        else:
            self.average_scores[self.metrics] = np.mean(self.scores[self.metrics])
        return self


class Task:
    def __init__(
        self,
        name: str,
        model: Model,
        labels,
        train_indices: List,
        val_indices: List,
        kfold=5,
        measure_performance=True,
        performance_measures=["accuracy"],
    ):
        """
        Parent class for the prediction task that is performed on top of the aligned representation
        :param name: Name of the task
        :param model: ML-Model to run
        :param labels: Labels used for prediction
        :param train_indices: Indices to extract training data
        :param val_indices: Indices to extract validation data
        :param kfold: Number of crossvalidation runs

        """
        self.name = name
        self.model = model
        self.labels = labels
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.kfold = kfold
        self.measure_performance = measure_performance
        self.inference_time = []
        self.training_time = []
        self.expected_dim = 1
        self.performance_measures = performance_measures
        self.train_scores = PerformanceMeasure("train", performance_measures)
        self.val_scores = PerformanceMeasure("val", performance_measures)

    def create_model(self):
        """
        Return a fresh, unfitted model instance.
        """
        if self.model is None:
            return None

        return copy.deepcopy(self.model)

    def get_train_test_split(self, data):
        X_train = [data[i] for i in self.train_indices]
        y_train = [self.labels[i] for i in self.train_indices]
        if self.val_indices is None:
            X_test = None
            y_test = None
        else:
            X_test = [data[i] for i in self.val_indices]
            y_test = [self.labels[i] for i in self.val_indices]

        return X_train, y_train, X_test, y_test

    def run(self, data):
        """
        The run method needs to be implemented by every task class
         It handles the training and validation procedures for the specific task
         :param data: The aligned data used in the prediction process
         :return: the validation accuracy
        """
        self._reset_params()
        model = self.create_model()
        skf = KFold(n_splits=self.kfold, shuffle=True, random_state=11)

        fold = 0
        X, y, _, _ = self.get_train_test_split(data)

        for train, test in skf.split(X, y):
            train_X = np.array(X)[train]
            train_y = np.array(y)[train]
            test_X = np.array(X)[test]
            test_y = np.array(y)[test]
            self._run_fold(model, train_X, train_y, test_X, test_y)
            fold += 1

        return [
            self.train_scores.compute_averages(),
            self.val_scores.compute_averages(),
        ]

    def _reset_params(self):
        self.inference_time = []
        self.training_time = []
        self.train_scores = PerformanceMeasure("train", self.performance_measures)
        self.val_scores = PerformanceMeasure("val", self.performance_measures)

    def _run_fold(self, model, train_X, train_y, test_X, test_y):
        train_start = time.time()
        train_score = model.fit(train_X, train_y, test_X, test_y)
        train_end = time.time()
        self.training_time.append(train_end - train_start)
        self.train_scores.add_scores(train_score[0])
        test_start = time.time()
        test_score = model.test(np.array(test_X), test_y)
        test_end = time.time()
        self.inference_time.append(test_end - test_start)
        self.val_scores.add_scores(test_score[0])

    def create_representation_and_run(
        self,
        representation: Representation,
        modalities: Union[List[Modality], Modality],
    ):
        self._reset_params()
        skf = KFold(n_splits=self.kfold, shuffle=True, random_state=11)

        fold = 0
        X, y, _, _ = self.get_train_test_split(data)

        for train, test in skf.split(X, y):
            train_X = np.array(X)[train]
            train_y = np.array(y)[train]
            test_X = s.transform(np.array(X)[test])
            test_y = np.array(y)[test]

            if isinstance(modalities, Modality):
                rep = modality.apply_representation(representation())
            else:
                representation().transform(
                    train_X, train_y
                )  # TODO: think about a way how to handle masks

            self._run_fold(train_X, train_y, test_X, test_y)
            fold += 1

        if self.measure_performance:
            self.inference_time = np.mean(self.inference_time)
            self.training_time = np.mean(self.training_time)

        return [np.mean(train_scores), np.mean(test_scores)]
