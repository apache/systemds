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
import copy
import numpy as np

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.utils import pad_sequences

from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.representation import RepresentationStats

from systemds.scuro.drsearch.operator_registry import register_fusion_operator


@register_fusion_operator()
class Concatenation(Fusion):
    def __init__(self, params=None, preserve_leading_axis=False):
        super().__init__("Concatenation")
        if params is not None:
            preserve_leading_axis = params.get(
                "preserve_leading_axis", preserve_leading_axis
            )

        self.preserve_leading_axis = bool(preserve_leading_axis)
        self.preserves_leading_axis = self.preserve_leading_axis

    def get_current_parameters(self):
        current_params = super().get_current_parameters()
        current_params["preserve_leading_axis"] = self.preserve_leading_axis
        return current_params

    @staticmethod
    def _as_dense(modality):
        dtype = modality.metadata[0]["data_layout"]["type"]
        data = modality.data
        arr = (
            np.asarray(data, dtype=dtype) if not isinstance(data, np.ndarray) else data
        )
        if arr.dtype == object:
            instances = [np.asarray(instance, dtype=dtype) for instance in data]
            rest = tuple(
                max(i.shape[d] for i in instances) for d in range(instances[0].ndim)
            )
            arr = np.zeros((len(instances), *rest), dtype=dtype)
            for i, instance in enumerate(instances):
                arr[(i, *(slice(0, s) for s in instance.shape))] = instance
        return arr

    @staticmethod
    def _to_window_feature_matrix(arr):
        if arr.ndim == 1:
            return arr[:, None, None]
        if arr.ndim == 2:
            return arr[:, :, None]
        return arr.reshape(arr.shape[0], arr.shape[1], -1)

    @staticmethod
    def _flatten_feature_shape(shape):
        if len(shape) == 0:
            return (1, 1)
        if len(shape) == 1:
            return (shape[0], 1)
        return (shape[0], int(np.prod(shape[1:])))

    def _concat_on_leading_axis(self, modalities: List[Modality]):
        arrays = [
            self._to_window_feature_matrix(self._as_dense(modality))
            for modality in modalities
        ]

        num_windows = max(arr.shape[1] for arr in arrays)
        aligned = []
        for arr in arrays:
            if arr.shape[1] < num_windows:
                pad_width = [(0, 0)] * arr.ndim
                pad_width[1] = (0, num_windows - arr.shape[1])
                arr = np.pad(arr, pad_width=pad_width, mode="constant")
            aligned.append(arr)

        return np.concatenate(aligned, axis=-1)

    def execute(self, modalities: List[Modality]):
        if len(modalities) == 1:
            return np.asarray(
                modalities[0].data,
                dtype=modalities[0].metadata[0]["data_layout"]["type"],
            )

        if self.preserve_leading_axis:
            return self._concat_on_leading_axis(modalities)

        max_emb_size = self.get_max_embedding_size(modalities)
        size = len(modalities[0].data)

        if np.array(modalities[0].data).ndim > 2:
            data = np.zeros((size, max_emb_size, 0))
        else:
            data = np.zeros((size, 0))

        for modality in modalities:
            other_modality = copy.deepcopy(modality.data)
            data = np.concatenate(
                [
                    data,
                    np.asarray(
                        other_modality,
                        dtype=modality.metadata[0]["data_layout"]["type"],
                    ),
                ],
                axis=-1,
            )

        return np.array(data)

    def get_output_stats(self, input_stats_list) -> RepresentationStats:
        stats_list = self._fusion_input_stats(input_stats_list)
        if not stats_list:
            return RepresentationStats(0, (0,))

        num_instances = max(s.num_instances for s in stats_list)
        shapes = [tuple(s.output_shape) for s in stats_list]
        if self.preserve_leading_axis:
            shapes = [self._flatten_feature_shape(shape) for shape in shapes]
        rank = len(shapes[0])

        if rank >= 1 and all(len(shape) == rank for shape in shapes):
            leading = tuple(max(shape[d] for shape in shapes) for d in range(rank - 1))
            output_shape = (*leading, sum(shape[-1] for shape in shapes))
        else:
            output_shape = max(stats_list, key=self._stats_num_elements).output_shape

        output_shape_is_known = all(s.output_shape_is_known for s in stats_list)
        return RepresentationStats(num_instances, output_shape, output_shape_is_known)

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        stats_list = self._as_stats_list(input_stats)
        input_bytes = sum(self._stats_bytes(s) for s in stats_list)
        output_bytes = self._stats_bytes(self.get_output_stats(input_stats))

        raw_bytes = self._raw_input_bytes(input_stats)
        cpu_peak = (
            int((raw_bytes + 2 * input_bytes + output_bytes) * 1.1) + 8 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}
