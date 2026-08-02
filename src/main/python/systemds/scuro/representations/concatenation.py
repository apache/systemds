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
    def __init__(self, params=None):
        """
        Combines modalities using concatenation
        """
        super().__init__("Concatenation")

    def execute(self, modalities: List[Modality]):
        if len(modalities) == 1:
            return np.asarray(
                modalities[0].data,
                dtype=modalities[0].metadata[0]["data_layout"]["type"],
            )

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
        if isinstance(input_stats_list, RepresentationStats):
            return input_stats_list

        stats_list = list(input_stats_list)
        if not stats_list:
            return RepresentationStats(0, (0,))

        num_instances = stats_list[0].num_instances
        total_dim = sum(s.output_shape[-1] for s in stats_list)
        output_shape = (total_dim,)

        return RepresentationStats(num_instances, output_shape)

    def estimate_peak_memory_bytes(self, input_stats_list) -> dict:
        elem_size = np.dtype(np.float32).itemsize

        def stats_bytes(s: RepresentationStats) -> int:
            numel = int(np.prod(s.output_shape)) if len(s.output_shape) > 0 else 1
            return int(s.num_instances * numel * elem_size)

        current_output = 0
        peak = 0
        for s in input_stats_list:
            chunk = stats_bytes(s)
            new_output = current_output + chunk

            step_peak = current_output + chunk + new_output + chunk
            peak = max(peak, step_peak)
            current_output = new_output

        cpu_peak = int(peak * 1.15 + 16 * 1024 * 1024)
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}
