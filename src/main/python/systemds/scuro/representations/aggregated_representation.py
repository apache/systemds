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
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import (
    Representation,
    RepresentationStats,
)
from systemds.scuro.representations.aggregate import Aggregation
import time
import numpy as np


class AggregatedRepresentation(Representation):
    def __init__(self, aggregation="mean", target_dimensions=None, params=None):
        if params is not None:
            aggregation = params["aggregation_function_aggregation_function"]
            target_dimensions = params["target_dimensions"]
        parameters = {
            "aggregation": list(Aggregation().get_aggregation_functions()),
        }
        super().__init__("AggregatedRepresentation", parameters)
        self.aggregation = Aggregation(aggregation)
        self.self_contained = True
        self.target_dimensions = target_dimensions
        self.data_type = np.float32

    def get_output_shape(self, input_stats: RepresentationStats) -> RepresentationStats:
        if len(input_stats.output_shape) == 1 or len(input_stats.output_shape) == 2:
            return RepresentationStats(
                input_stats.num_instances, (input_stats.output_shape[0])
            )
        elif len(input_stats.output_shape) == 3:
            return RepresentationStats(
                input_stats.num_instances,
                (input_stats.output_shape[0], input_stats.output_shape[1]),
            )
        else:
            raise ValueError(f"Invalid output shape: {input_stats.output_shape}")

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        output_memory_bytes = 1
        output_shape = self.get_output_shape(input_stats).output_shape
        for dim in output_shape:
            output_memory_bytes *= dim
        return (
            input_stats.num_instances
            * output_memory_bytes
            * np.dtype(self.data_type).itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        return {
            "cpu_peak_bytes": self.estimate_output_memory_bytes(input_stats),
            "gpu_peak_bytes": 0,
        }

    def transform(self, modality):
        start = time.perf_counter()
        aggregated_modality = TransformedModality(
            modality, self, self_contained=modality.self_contained
        )
        if self.target_dimensions is not None:
            input_dimensions = modality.data[0].shape
            if len(input_dimensions) == self.target_dimensions:
                return modality
            else:
                while len(input_dimensions) > self.target_dimensions:
                    aggregated_data = self.aggregation.execute(modality)
                    input_dimensions = aggregated_data[0].shape

        else:
            aggregated_data = self.aggregation.execute(modality)

        aggregated_modality.data = aggregated_data
        end = time.perf_counter()
        aggregated_modality.transform_time += end - start
        return aggregated_modality

    def get_current_parameters(self):
        current_params = {}
        for key, value in self.aggregation.get_current_parameters().items():
            current_params[f"aggregation_function_{key}"] = value
        current_params["self_contained"] = self.self_contained
        current_params["target_dimensions"] = self.target_dimensions
        return current_params
