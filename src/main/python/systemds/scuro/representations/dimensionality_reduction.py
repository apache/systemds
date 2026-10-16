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

from systemds.scuro.modality.indexed import IndexedModality
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import (
    FitRequirement,
    Representation,
    RepresentationFitState,
)


class DimensionalityReduction(Representation):
    def __init__(self, name, params=None):
        """
        Parent class for different dimensionality reduction operations
        :param name: Name of the dimensionality reduction operator
        """
        super().__init__(name, params)
        self.needs_training = False

    @abc.abstractmethod
    def execute(self, data, labels=None):
        """
        Implemented for every child class and creates a sampled representation for a given modality
        :param data: data to apply the dimensionality reduction on
        :param labels: labels for learned dimensionality reduction
        :return: dimensionality reduced data
        """
        raise NotImplementedError(f"execute is not implemented for {self.name}")

    def apply_representation(self, data):
        """
        Implemented for every child class and creates a dimensionality reduced representation for a given modality
        :param data: data to apply the representation on
        :return: dimensionality reduced data
        """
        raise f"Not implemented for Dimensionality Reduction Operator: {self.name}"

    @staticmethod
    def _prepare_data(modality):
        data = np.asarray(modality.data)
        if data.ndim == 3 and data.shape[1] == 1:
            return data.reshape(data.shape[0], -1)
        return data

    def fit(self, inputs, labels=None) -> RepresentationFitState:
        if self.fit_requirement is not FitRequirement.SUPERVISED:
            return super().fit(inputs, labels)
        if not isinstance(inputs, IndexedModality):
            raise TypeError(
                "Trainable dimensionality reduction expects an IndexedModality"
            )
        if labels is None:
            raise ValueError(f"{self.name} requires labels for fitting")
        if len(labels) != len(inputs):
            raise ValueError(
                f"Expected {len(inputs)} labels for {self.name}, got {len(labels)}"
            )

        self.execute(self._prepare_data(inputs.modality), np.asarray(labels))
        return RepresentationFitState(
            self.fit_requirement,
            fit_instance_ids=inputs.instance_ids,
            payload=self,
        )

    def transform_fitted(self, inputs, fit_state, **kwargs):
        if not isinstance(inputs, IndexedModality):
            raise TypeError(
                "Trainable dimensionality reduction expects an IndexedModality"
            )
        if (
            not isinstance(fit_state, RepresentationFitState)
            or fit_state.fit_requirement is not self.fit_requirement
            or not isinstance(fit_state.payload, type(self))
        ):
            raise ValueError(f"{self.name} requires a compatible fitted state")

        transformed = TransformedModality(inputs.modality, fit_state.payload)
        transformed.data = fit_state.payload.apply_representation(
            self._prepare_data(inputs.modality)
        )
        return IndexedModality(transformed, inputs.instance_ids)
