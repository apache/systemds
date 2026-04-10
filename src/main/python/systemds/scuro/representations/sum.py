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
from systemds.scuro.representations.utils import pad_sequences

from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.representation import RepresentationStats

from systemds.scuro.drsearch.operator_registry import register_fusion_operator


@register_fusion_operator()
class Sum(Fusion):
    def __init__(self, params=None):
        """
        Combines modalities using colum-wise sum
        """
        super().__init__("Sum")
        self.needs_alignment = True

    def execute(self, modalities: List[Modality]):
        data = np.asarray(
            modalities[0].data,
            dtype=modalities[0].metadata[list(modalities[0].metadata.keys())[0]][
                "data_layout"
            ]["type"],
        )

        for m in range(1, len(modalities)):
            data += np.asarray(
                modalities[m].data,
                dtype=modalities[m].metadata[list(modalities[m].metadata.keys())[0]][
                    "data_layout"
                ]["type"],
            )
        return data

    def get_output_stats(self, input_stats_list) -> RepresentationStats:
        if isinstance(input_stats_list, RepresentationStats):
            return input_stats_list

        stats_list = list(input_stats_list)
        if not stats_list:
            return RepresentationStats(0, (0,))

        def num_elements(stats: RepresentationStats) -> int:
            n = 1
            for d in stats.output_shape:
                n *= d
            return n

        largest = max(stats_list, key=num_elements)
        return RepresentationStats(largest.num_instances, largest.output_shape)

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        # TODO
        return {
            "cpu_peak_bytes": 0,
            "gpu_peak_bytes": 0,
        }
