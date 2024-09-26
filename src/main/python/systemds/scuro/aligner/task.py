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
from typing import List

from systemds.scuro.models.model import Model


class Task:
    def __init__(
        self, name: str, model: Model, labels, train_indices: List, val_indices: List
    ):
        """
        Parent class for the prediction task that is performed on top of the aligned representation
        :param name: Name of the task
        :param model: ML-Model to run
        :param labels: Labels used for prediction
        :param train_indices: Indices to extract training data
        :param val_indices: Indices to extract validation data
        """
        self.name = name
        self.model = model
        self.labels = labels
        self.train_indices = train_indices
        self.val_indices = val_indices

    def get_train_test_split(self, data):
        X_train = [data[i] for i in self.train_indices]
        y_train = [self.labels[i] for i in self.train_indices]
        X_test = [data[i] for i in self.val_indices]
        y_test = [self.labels[i] for i in self.val_indices]

        return X_train, y_train, X_test, y_test

    def run(self, data):
        """
        The run method need to be implemented by every task class
        It handles the training and validation procedures for the specific task
        :param data: The aligned data used in the prediction process
        :return: the validation accuracy
        """
        pass
