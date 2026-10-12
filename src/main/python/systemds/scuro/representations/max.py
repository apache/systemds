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
from systemds.scuro.representations.fusion import Fusion

from systemds.scuro.drsearch.operator_registry import register_fusion_operator


@register_fusion_operator()
class RowMax(Fusion):
    def __init__(self, params=None):
        """
        Combines modalities by computing the outer product of a modality combination and
        taking the row max
        """
        super().__init__("RowMax")
        self.needs_alignment = True
        self.associative = True
        self.commutative = True

    def execute(
        self,
        modalities: List[Modality],
    ):
        fused_data = np.maximum.reduce([m.data for m in modalities])
        return fused_data

    def get_output_stats(self, input_stats_list) -> RepresentationStats:
        # Elementwise over operands aligned to the widest, as in Average.
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
