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
import time
from typing import List

from systemds.scuro.models.model import Model
import numpy as np
from sklearn.model_selection import KFold


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

    def get_train_test_split(self, data):
        X_train = [data[i] for i in self.train_indices]
        y_train = [self.labels[i] for i in self.train_indices]
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
        self.inference_time = []
        self.training_time = []
        skf = KFold(n_splits=self.kfold, shuffle=True, random_state=11)
        train_scores = []
        test_scores = []
        fold = 0
        X, y, X_test, y_test = self.get_train_test_split(data)

        for train, test in skf.split(X, y):
            train_X = np.array(X)[train]
            train_y = np.array(y)[train]
            train_start = time.time()
            train_score = self.model.fit(train_X, train_y, X_test, y_test)
            train_end = time.time()
            self.training_time.append(train_end - train_start)
            train_scores.append(train_score)
            test_start = time.time()
            test_score = self.model.test(np.array(X_test), y_test)
            test_end = time.time()
            self.inference_time.append(test_end - test_start)
            test_scores.append(test_score)

            fold += 1

        if self.measure_performance:
            self.inference_time = np.mean(self.inference_time)
            self.training_time = np.mean(self.training_time)

        return [np.mean(train_scores), np.mean(test_scores)]
