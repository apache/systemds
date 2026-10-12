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
from typing import List, Optional

import numpy as np

from systemds.scuro.modality.indexed import IndexedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.modality.transformed import TransformedModality

from systemds.scuro.modality.modality import Modality, selective_copy_metadata
from systemds.scuro.representations.representation import (
    CONTAINER_ARRAY,
    FitRequirement,
    Representation,
    RepresentationFitState,
    RepresentationStats,
    derive_stats,
    stats_bytes,
    stats_num_elements,
)
from systemds.scuro.utils.schema_helpers import get_shape

DEFAULT_MAX_ALIGNMENT_PAD_RATIO = None


class FusionInputTooLargeError(ValueError):
    """The operands are too wide for this fusion operator's model.

    Raised before any training, so a search can discard the candidate as
    oversized instead of counting it as a failed evaluation.
    """


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
        self.max_alignment_pad_ratio = DEFAULT_MAX_ALIGNMENT_PAD_RATIO

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

    def input_size_violation(self, input_stats) -> Optional[str]:
        """Why these operands are too wide for this operator, or None.

        Predicted from shapes alone so that a search can reject a candidate
        before executing it. Operators without an input-width limit accept
        everything.
        """
        return None

    def _operand_widths(self, input_stats) -> Optional[List[int]]:
        """Per-operand feature width as _prepare_modalities hands it to execute.

        The width is the last axis of each operand after pre-aggregation;
        operators that need alignment zero-pad every operand to the widest.
        None if a shape is unknown or has more than two axes.
        """
        widths = []
        for stats in self._fusion_input_stats(input_stats):
            shape = tuple(stats.output_shape or ())
            if not stats.output_shape_is_known or len(shape) not in (1, 2):
                return None
            widths.append(int(shape[-1]))
        if self.needs_alignment and widths:
            widths = [max(widths)] * len(widths)
        return widths

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
        mods = self._prepare_modalities(modalities)

        return self.execute(mods)

    def _prepare_modalities(self, modalities: List[Modality]):
        mods = []
        for modality in modalities:
            agg_modality = None
            if (
                not self.preserves_leading_axis
                and modality.has_metadata()
                and get_shape(modality.metadata) > 1
            ):
                agg_operator = AggregatedRepresentation(
                    target_dimensions=1, aggregate_leading=True
                )
                agg_modality = agg_operator.transform(modality)
            mods.append(agg_modality if agg_modality else modality)

        if self.needs_alignment:
            mods = [self._detached_copy(modality) for modality in mods]
            for modality in mods:
                self._normalize_for_fusion(modality)
            max_len = self.get_max_embedding_size(mods)
            self._check_alignment_padding(mods, max_len)
            for modality in mods:
                modality.pad(max_len=max_len)

        return mods

    @staticmethod
    def _detached_copy(modality: Modality) -> Modality:
        detached = copy.copy(modality)
        detached.metadata = selective_copy_metadata(modality.metadata)
        return detached

    def _check_alignment_padding(self, modalities: List[Modality], max_len: int):
        if not self.max_alignment_pad_ratio or max_len <= 0:
            return

        for modality in modalities:
            width = self._embedding_width(modality)
            if width <= 0:
                continue
            if max_len > width * self.max_alignment_pad_ratio:
                raise ValueError(
                    f"{self.name} would have to zero-pad modality "
                    f"{modality.modality_id} from {width:,} to {max_len:,} "
                    f"features to align it, exceeding the maximum padding ratio "
                    f"of {self.max_alignment_pad_ratio:g}x. Aggregate or reduce "
                    "the operands to comparable widths before fusing them "
                    "elementwise."
                )

    @staticmethod
    def _validate_indexed_modalities(inputs):
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise TypeError(
                "Fusion fitting expects a non-empty sequence of IndexedModality inputs"
            )
        if not all(isinstance(item, IndexedModality) for item in inputs):
            raise TypeError("Fusion fitting expects IndexedModality inputs")

        instance_ids = inputs[0].instance_ids
        if any(item.instance_ids != instance_ids for item in inputs[1:]):
            raise ValueError("Fusion inputs must have aligned instance IDs")
        return instance_ids

    def fit(self, inputs, labels=None) -> RepresentationFitState:
        if self.fit_requirement is not FitRequirement.SUPERVISED:
            return super().fit(inputs, labels)
        instance_ids = self._validate_indexed_modalities(inputs)
        if labels is None:
            raise ValueError(f"{self.name} requires labels for fitting")
        if len(labels) != len(instance_ids):
            raise ValueError(
                f"Expected {len(instance_ids)} labels for {self.name}, got {len(labels)}"
            )

        modalities = self._prepare_modalities([item.modality for item in inputs])
        self.execute(modalities, np.asarray(labels))
        return RepresentationFitState(
            self.fit_requirement,
            fit_instance_ids=instance_ids,
            payload=self,
        )

    def transform_fitted(self, inputs, fit_state, **kwargs):
        instance_ids = self._validate_indexed_modalities(inputs)
        if (
            not isinstance(fit_state, RepresentationFitState)
            or fit_state.fit_requirement is not self.fit_requirement
            or not isinstance(fit_state.payload, type(self))
        ):
            raise ValueError(f"{self.name} requires a compatible fitted state")

        modalities = fit_state.payload._prepare_modalities(
            [item.modality for item in inputs]
        )
        transformed = TransformedModality(
            modalities[0], fit_state.payload, ModalityType.EMBEDDING
        )
        transformed.data = fit_state.payload.apply_representation(modalities)
        return IndexedModality(transformed, instance_ids)

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

    @staticmethod
    def _embedding_width(m: Modality) -> int:
        data = m.data
        if isinstance(data, memoryview):
            data = np.array(data)
        arr = np.asarray(data)
        if arr.dtype == object:
            return 0
        if arr.ndim == 1:
            if m.has_metadata() and len(m.metadata) == 1:
                return arr.shape[0]
            if not m.has_metadata() or len(m.metadata) <= 1:
                return arr.shape[0]
            return 0
        if arr.ndim >= 2:
            return arr.shape[1]
        return 0

    def get_max_embedding_size(self, modalities: List[Modality]):
        """
        Computes the maximum embedding size from a given list of modalities
        :param modalities: List of modalities
        :return: maximum embedding size
        """

        return max((self._embedding_width(m) for m in modalities), default=0)
