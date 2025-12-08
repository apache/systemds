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
from typing import List

import numpy as np
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.modality.transformed import TransformedModality

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.representation import Representation
from systemds.scuro.utils.schema_helpers import get_shape


class Fusion(Representation):
    def __init__(self, name, parameters=None):
        """
        Parent class for different multimodal fusion types
        :param name: Name of the fusion type
        """
        super().__init__(name, parameters)
        self.associative = False
        self.commutative = False
        self.needs_alignment = False
        self.needs_training = False
        self.needs_instance_alignment = False

    def transform(self, modalities: List[Modality]):
        """
        Implemented for every child class and creates a fused representation out of
        multiple modalities
        :param modalities: List of modalities used in the fusion
        :return: fused data
        """
        mods = []
        for modality in modalities:
            agg_modality = None
            if get_shape(modality.metadata) > 1:
                agg_operator = AggregatedRepresentation()
                agg_modality = agg_operator.transform(modality)
            mods.append(agg_modality if agg_modality else modality)

        if self.needs_alignment:
            max_len = self.get_max_embedding_size(mods)
            for modality in mods:
                modality.pad(max_len=max_len)

        return self.execute(mods)

    def transform_with_training(self, modalities: List[Modality], task):
        train_modalities = []
        for modality in modalities:
            train_data = [
                d for i, d in enumerate(modality.data) if i in task.train_indices
            ]
            train_modality = TransformedModality(modality, self)
            train_modality.data = copy.deepcopy(train_data)
            train_modalities.append(train_modality)

        transformed_train = self.execute(
            train_modalities, task.labels[task.train_indices]
        )
        transformed_val = self.transform_data(modalities, task.val_indices)

        transformed_data = np.zeros(
            (len(modalities[0].data), transformed_train.shape[1])
        )
        transformed_data[task.train_indices] = transformed_train
        transformed_data[task.test_indices] = transformed_other

        return transformed_data

    def transform_data(self, modalities: List[Modality], indices=None):
        val_modalities = []
        for modality in modalities:
            val_data = (
                [d for i, d in enumerate(modality.data) if i in indices]
                if indices
                else modality.data
            )
            val_modality = type(modality)(modality, self)
            val_modality.data = copy.deepcopy(val_data)
            val_modalities.append(val_modality)

        return self.apply_representation(val_modalities)

    def execute(self, modalities: List[Modality], labels: np.ndarray = None):
        raise NotImplementedError(f"Not implemented for Fusion: {self.name}")

    def apply_representation(self, modalities: List[Modality]):
        if self.needs_training:
            raise NotImplementedError(
                f"apply_representation not implemented for trainable fusion: {self.name}"
            )
        else:
            return self.execute(modalities)

    def get_max_embedding_size(self, modalities: List[Modality]):
        """
        Computes the maximum embedding size from a given list of modalities
        :param modalities: List of modalities
        :return: maximum embedding size
        """

        max_size = 0
        for m in modalities:
            data = m.data
            if isinstance(data, memoryview):
                data = np.array(data)
            arr = np.asarray(data)
            if arr.ndim < 2:
                continue
            emb_size = arr.shape[1]
            if emb_size > max_size:
                max_size = emb_size
        return max_size
