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
from typing import List
from systemds.scuro.models.model import Model
import numpy as np
from sklearn.model_selection import train_test_split


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
        fusion_train_split=0.8,
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
        self.test_indices = val_indices
        self.kfold = kfold
        self.measure_performance = measure_performance
        self.inference_time = []
        self.training_time = []
        self.expected_dim = 1
        self.performance_measures = performance_measures
        self.train_scores = PerformanceMeasure("train", performance_measures)
        self.val_scores = PerformanceMeasure("val", performance_measures)
        self.test_scores = PerformanceMeasure("test", performance_measures)
        self.fusion_train_indices = None
        self._create_cv_splits()

    def _create_cv_splits(self):
        train_labels = [self.labels[i] for i in self.train_indices]
        train_labels_array = np.array(train_labels)

        train_indices_array = np.array(self.train_indices)

        self.cv_train_indices = []
        self.cv_val_indices = []

        for fold_idx in range(self.kfold):
            fold_train_indices_array, fold_val_indices_array, _, _ = train_test_split(
                train_indices_array,
                train_labels_array,
                test_size=0.2,
                shuffle=True,
                random_state=11 + fold_idx,
            )

            fold_train_indices = fold_train_indices_array.tolist()
            fold_val_indices = fold_val_indices_array.tolist()

            self.cv_train_indices.append(fold_train_indices)
            self.cv_val_indices.append(fold_val_indices)

            overlap = set(fold_train_indices) & set(fold_val_indices)
            if overlap:
                raise ValueError(
                    f"Fold {fold_idx}: Overlap detected between train and val indices: {overlap}"
                )

        all_val_indices = set()
        for val_indices in self.cv_val_indices:
            all_val_indices.update(val_indices)

        self.fusion_train_indices = [
            idx for idx in self.train_indices if idx not in all_val_indices
        ]

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
        if self.test_indices is None:
            X_test = None
            y_test = None
        else:
            X_test = [data[i] for i in self.test_indices]
            y_test = [self.labels[i] for i in self.test_indices]

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

        test_X = np.array([data[i] for i in self.test_indices])
        test_y = np.array([self.labels[i] for i in self.test_indices])

        for fold_idx in range(self.kfold):
            fold_train_indices = self.cv_train_indices[fold_idx]
            fold_val_indices = self.cv_val_indices[fold_idx]

            train_X = np.array([data[i] for i in fold_train_indices])
            train_y = np.array([self.labels[i] for i in fold_train_indices])
            val_X = np.array([data[i] for i in fold_val_indices])
            val_y = np.array([self.labels[i] for i in fold_val_indices])

            self._run_fold(model, train_X, train_y, val_X, val_y, test_X, test_y)

        return [
            self.train_scores.compute_averages(),
            self.val_scores.compute_averages(),
            self.test_scores.compute_averages(),
        ]

    def _reset_params(self):
        self.inference_time = []
        self.training_time = []
        self.train_scores = PerformanceMeasure("train", self.performance_measures)
        self.val_scores = PerformanceMeasure("val", self.performance_measures)
        self.test_scores = PerformanceMeasure("test", self.performance_measures)

    def _run_fold(self, model, train_X, train_y, val_X, val_y, test_X, test_y):
        train_start = time.time()
        train_score = model.fit(train_X, train_y, val_X, val_y)
        train_end = time.time()
        self.training_time.append(train_end - train_start)
        self.train_scores.add_scores(train_score[0])
        val_score = model.test(val_X, val_y)
        test_start = time.time()
        test_score = model.test(np.array(test_X), test_y)
        test_end = time.time()
        self.inference_time.append(test_end - test_start)
        self.val_scores.add_scores(val_score[0])
        self.test_scores.add_scores(test_score[0])
