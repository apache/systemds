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
            if "aggregation_function_aggregation_function" in params:
                aggregation = params["aggregation_function_aggregation_function"]
            elif "aggregation_function" in params:
                aggregation = params["aggregation_function"]
            else:
                aggregation = params["aggregation"]
            if "target_dimensions" in params:
                target_dimensions = params["target_dimensions"]
        parameters = {
            "aggregation": list(Aggregation().get_aggregation_functions()),
        }
        super().__init__("AggregatedRepresentation", parameters)
        self.aggregation_function = aggregation
        self.aggregation = Aggregation(aggregation)
        self.self_contained = True
        self.target_dimensions = target_dimensions
        self.data_type = np.float32

    def get_output_stats(self, input_stats: RepresentationStats) -> RepresentationStats:
        input_shape = list(copy.deepcopy(input_stats.output_shape))
        input_aggregate_dim = copy.deepcopy(input_stats.aggregate_dim)
        for input_aggregate_dim in reversed(input_aggregate_dim):
            input_shape.pop(input_aggregate_dim)
        out_shape = tuple(input_shape)
        self.stats = RepresentationStats(
            input_stats.num_instances,
            out_shape,
            output_shape_is_known=input_stats.output_shape_is_known,
            aggregate_dim=input_stats.aggregate_dim,
        )
        return self.stats

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        out_shape = self.get_output_stats(input_stats).output_shape
        out_numel = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
        dtype_size = np.dtype(self.data_type).itemsize
        return int(input_stats.num_instances * out_numel * dtype_size)

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        dtype_size = np.dtype(self.data_type).itemsize
        in_shape = tuple(input_stats.output_shape)
        in_numel = int(np.prod(in_shape)) if len(in_shape) > 0 else 1
        input_bytes = int(input_stats.num_instances * in_numel * dtype_size)
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        safety = 1.2
        cpu_peak = input_bytes * 2 + output_bytes * 2

        return {
            "cpu_peak_bytes": int(cpu_peak * safety),
            "gpu_peak_bytes": 0,
        }

    def execute(self, data):
        return self.aggregation.compute_feature(data)

    def transform(self, modality):
        start = time.perf_counter()
        aggregated_modality = TransformedModality(
            modality, self, self_contained=modality.self_contained
        )

        aggregate_dim = (0,)
        if self.target_dimensions is not None:
            input_dimensions = self._get_input_dimensions(modality.data)

            if len(input_dimensions) == self.target_dimensions:
                return modality
            else:

                i = 1
                while len(input_dimensions) - 1 > self.target_dimensions:
                    aggregate_dim = aggregate_dim + (i,)
                    i += 1
                    input_dimensions = input_dimensions[:-1]

        aggregated_data = self.aggregation.execute(modality, aggregate_dim)

        aggregated_modality.data = aggregated_data
        end = time.perf_counter()
        aggregated_modality.transform_time += end - start

        self.assert_output_stats(aggregated_data)
        return aggregated_modality

    def get_current_parameters(self):
        current_params = {}
        for key, value in self.aggregation.get_current_parameters().items():
            current_params[f"aggregation_function_{key}"] = value
        current_params["self_contained"] = self.self_contained
        current_params["target_dimensions"] = self.target_dimensions
        return current_params

    def assert_output_stats(self, aggregated_data):
        if self.stats:
            assert len(aggregated_data) == self.stats.num_instances
            assert aggregated_data[0].shape == self.stats.output_shape

    def _get_input_dimensions(self, data):
        if isinstance(data[0], list):
            input_dimensions = (len(data[0]), *data[0][0].shape)
        else:
            input_dimensions = data[0].shape
        return input_dimensions
