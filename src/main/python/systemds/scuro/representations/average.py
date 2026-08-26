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

import numpy as np

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.utils import pad_sequences

from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.drsearch.operator_registry import register_fusion_operator


@register_fusion_operator()
class Average(Fusion):
    def __init__(self, params=None):
        """
        Combines modalities using averaging
        """
        super().__init__("Average")
        self.needs_alignment = True
        self.associative = True
        self.commutative = True

    def execute(self, modalities: List[Modality]):
        data = np.array(modalities[0].data, dtype=np.float64)
        for i in range(1, len(modalities)):
            data += np.asarray(modalities[i].data, dtype=np.float64)

        data /= len(modalities)

        return data

    def get_output_stats(self, input_stats_list) -> RepresentationStats:
        stats_list = self._fusion_input_stats(input_stats_list)
        if not stats_list:
            return RepresentationStats(0, (0,))

        num_instances = max(s.num_instances for s in stats_list)
        rank = len(stats_list[0].output_shape)
        if rank > 0 and all(len(s.output_shape) == rank for s in stats_list):
            output_shape = tuple(
                max(s.output_shape[d] for s in stats_list) for d in range(rank)
            )
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
            int((raw_bytes + input_bytes + 2 * output_bytes) * 1.15) + 8 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}
