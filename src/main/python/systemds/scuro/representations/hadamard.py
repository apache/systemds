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

from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.representation import RepresentationStats

from systemds.scuro.drsearch.operator_registry import register_fusion_operator


@register_fusion_operator()
class Hadamard(Fusion):
    def __init__(self, params=None):
        """
        Combines modalities using elementwise multiply (Hadamard product)
        """
        super().__init__("Hadamard")
        self.needs_alignment = True  # zero padding falsifies the result
        self.commutative = True
        self.associative = True

    def execute(self, modalities: List[Modality], train_indices=None):
        fused_data = np.prod([m.data for m in modalities], axis=0)

        return fused_data

    def get_output_stats(self, input_stats_list) -> RepresentationStats:
        if isinstance(input_stats_list, RepresentationStats):
            return input_stats_list

        stats_list = list(input_stats_list)
        if not stats_list:
            return RepresentationStats(0, (0,))

        max_dim = max([stats.output_shape[-1] for stats in stats_list])
        return RepresentationStats(stats_list[0].num_instances, (max_dim,))

    def estimate_peak_memory_bytes(self, input_stats_list) -> dict:
        elem_size = np.dtype(np.float64).itemsize

        def stats_payload_bytes(s: RepresentationStats) -> int:
            numel = int(np.prod(s.output_shape)) if len(s.output_shape) > 0 else 1
            return int(s.num_instances * numel * elem_size)

        stacked_input_bytes = sum(stats_payload_bytes(s) for s in input_stats_list)
        out_stats = self.get_output_stats(input_stats_list)
        output_bytes = stats_payload_bytes(out_stats)
        reduction_workspace_bytes = output_bytes
        cpu_peak = int(
            (stacked_input_bytes + output_bytes + reduction_workspace_bytes) * 1.15
            + 8 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}
