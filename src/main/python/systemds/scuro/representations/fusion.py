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

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.modality.transformed import TransformedModality

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.representation import (
    CONTAINER_ARRAY,
    Representation,
    RepresentationStats,
    derive_stats,
    stats_bytes,
    stats_num_elements,
)
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
        self.preserves_leading_axis = False
        self.output_modality_type = ModalityType.EMBEDDING
        self.data_type = np.float32

    @staticmethod
    def _as_stats_list(input_stats) -> List[RepresentationStats]:
        if isinstance(input_stats, RepresentationStats):
            return [input_stats]
        return list(input_stats)

    def _pre_aggregated_stats(self, stats: RepresentationStats) -> RepresentationStats:
        shape = tuple(stats.output_shape)
        if len(shape) > 1 and not self.preserves_leading_axis:
            shape = shape[1:]

        return derive_stats(
            stats,
            output_shape=shape,
            dtype=self.data_type,
            container=CONTAINER_ARRAY,
        )

    def _fusion_input_stats(self, input_stats) -> List[RepresentationStats]:
        return [self._pre_aggregated_stats(s) for s in self._as_stats_list(input_stats)]

    def _raw_input_bytes(self, input_stats) -> int:
        return sum(
            stats_bytes(s, quantile=0.95) for s in self._as_stats_list(input_stats)
        )

    @staticmethod
    def _stats_num_elements(stats: RepresentationStats) -> int:
        return stats_num_elements(stats)

    def _stats_bytes(self, stats: RepresentationStats) -> int:
        return stats_bytes(stats)

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
            if not self.preserves_leading_axis and get_shape(modality.metadata) > 1:
                agg_operator = AggregatedRepresentation()
                agg_modality = agg_operator.transform(modality)
            mods.append(agg_modality if agg_modality else modality)

        if self.needs_alignment:
            for modality in mods:
                self._normalize_for_fusion(modality)
            max_len = self.get_max_embedding_size(mods)
            for modality in mods:
                modality.pad(max_len=max_len)

        return self.execute(mods)

    def transform_with_training(self, modalities: List[Modality], task):
        fusion_train_indices = task.fusion_train_indices

        train_modalities = []
        for modality in modalities:
            train_data = [
                d for i, d in enumerate(modality.data) if i in fusion_train_indices
            ]
            train_modality = TransformedModality(modality, self)
            train_modality.data = list(train_data)
            train_modalities.append(train_modality)

        transformed_train = self.execute(
            train_modalities, task.labels[fusion_train_indices]
        )

        all_other_indices = [
            i for i in range(len(modalities[0].data)) if i not in fusion_train_indices
        ]
        transformed_other = self.transform_data(modalities, all_other_indices)

        transformed_data = np.zeros(
            (len(modalities[0].data), transformed_train.shape[1])
        )
        transformed_data[fusion_train_indices] = transformed_train
        transformed_data[all_other_indices] = transformed_other

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

    @staticmethod
    def _normalize_for_fusion(modality: Modality):
        if not modality.has_data():
            return

        arr = np.asarray(modality.data)
        if arr.dtype == object or arr.ndim != 1:
            return

        if modality.has_metadata() and len(modality.metadata) == 1:
            modality.data = arr.reshape(1, -1).copy()
        elif not modality.has_metadata() or len(modality.metadata) <= 1:
            modality.data = arr.reshape(1, -1).copy()

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
            if arr.dtype == object:
                continue
            if arr.ndim == 1:
                if m.has_metadata() and len(m.metadata) == 1:
                    emb_size = arr.shape[0]
                elif not m.has_metadata() or len(m.metadata) <= 1:
                    emb_size = arr.shape[0]
                else:
                    continue
            elif arr.ndim >= 2:
                emb_size = arr.shape[1]
            else:
                continue
            if emb_size > max_size:
                max_size = emb_size
        return max_size
