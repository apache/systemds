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
import abc

import numpy as np

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.representation import Representation


class DimensionalityReduction(Representation):
    def __init__(self, name, parameters=None):
        """
        Parent class for different dimensionality reduction operations
        :param name: Name of the dimensionality reduction operator
        """
        super().__init__(name, parameters)
        self.needs_training = False

    @abc.abstractmethod
    def execute(self, data, labels=None):
        """
        Implemented for every child class and creates a sampled representation for a given modality
        :param data: data to apply the dimensionality reduction on
        :param labels: labels for learned dimensionality reduction
        :return: dimensionality reduced data
        """
        if labels is not None:
            self.execute_with_training(data, labels)
        else:
            self.execute(data)

    def apply_representation(self, data):
        """
        Implemented for every child class and creates a dimensionality reduced representation for a given modality
        :param data: data to apply the representation on
        :return: dimensionality reduced data
        """
        raise f"Not implemented for Dimensionality Reduction Operator: {self.name}"

    def execute_with_training(self, modality, task):
        fusion_train_indices = task.fusion_train_indices
        # Handle 3d data
        data = modality.data
        if (
            len(np.array(modality.data).shape) == 3
            and np.array(modality.data).shape[1] == 1
        ):
            data = np.array([x.reshape(-1) for x in modality.data])
        transformed_train = self.execute(
            np.array(data)[fusion_train_indices], task.labels[fusion_train_indices]
        )

        all_other_indices = [
            i for i in range(len(modality.data)) if i not in fusion_train_indices
        ]
        transformed_other = self.apply_representation(np.array(data)[all_other_indices])

        transformed_data = np.zeros((len(data), transformed_train.shape[1]))
        transformed_data[fusion_train_indices] = transformed_train
        transformed_data[all_other_indices] = transformed_other

        return transformed_data
