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
import numpy as np

from systemds.scuro.dataloader.tabular_loader import TabularStats
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation


@register_representation(ModalityType.EMBEDDING)
class TabularFeatures(UnimodalRepresentation):
    def __init__(self, params=None):
        super().__init__("TabularFeatures", ModalityType.EMBEDDING, None)
        self.data_type = np.float32

    def get_output_stats(self, input_stats: TabularStats) -> RepresentationStats:
        return RepresentationStats(
            input_stats.num_instances, input_stats.output_shape, dtype=self.data_type
        )

    def estimate_output_memory_bytes(self, input_stats: TabularStats) -> int:
        return (
            input_stats.num_instances
            * input_stats.num_features
            * np.dtype(self.data_type).itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats: TabularStats) -> dict:
        return {
            "cpu_peak_bytes": self.estimate_output_memory_bytes(input_stats) * 2,
            "gpu_peak_bytes": 0,
        }

    def transform(self, modality, params=None):
        transformed_modality = TransformedModality(modality, self)
        transformed_modality.data_type = self.data_type
        transformed_modality.data = np.array(modality.data, dtype=self.data_type)
        return transformed_modality
