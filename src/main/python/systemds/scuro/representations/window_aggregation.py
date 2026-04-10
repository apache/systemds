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
import math

from systemds.scuro.modality.type import DataLayout, ModalityType

from systemds.scuro.drsearch.operator_registry import register_context_operator
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.context import Context
from systemds.scuro.representations.representation import (
    Representation,
    RepresentationStats,
)


class Window(Context):
    def __init__(self, name, aggregation_function):
        self.aggregation_function = aggregation_function
        parameters = {}
        if isinstance(self.aggregation_function, Aggregation) or isinstance(
            self.aggregation_function, Representation
        ):
            parameters["aggregation_function"] = self.aggregation_function.__class__
        else:
            raise ValueError(
                f"Invalid aggregation function type: {type(self.aggregation_function)}"
            )

        super().__init__(name, parameters)
        self.data_type = np.float32

    def get_current_parameters(self):
        current_params = {}
        if not self.parameters:
            return current_params

        for parameter in list(self.parameters.keys()):
            if parameter == "aggregation_function":
                if "aggregation_function" in self.parameters:
                    current_params["aggregation_function"] = self.parameters[
                        "aggregation_function"
                    ]
                    if isinstance(self.aggregation_function, Aggregation) or isinstance(
                        self.aggregation_function, Representation
                    ):
                        for (
                            key,
                            value,
                        ) in self.aggregation_function.get_current_parameters().items():
                            current_params[f"aggregation_function_{key}"] = value
            else:
                current_params[parameter] = getattr(self, parameter)
        return current_params

    @property
    def aggregation_function(self):
        return self._aggregation_function

    @aggregation_function.setter
    def aggregation_function(self, value):
        if isinstance(value, Representation):
            self._aggregation_function = value
        elif isinstance(value, Aggregation):
            self._aggregation_function = value
        else:
            self._aggregation_function = Aggregation(value)

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        return None

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        return None

    def assert_output_stats(self, windowed_data):
        if self.stats:
            assert (
                len(windowed_data) == self.stats.num_instances
            ), f"Output shape: {windowed_data.shape}, Expected shape: {self.stats.output_shape}"
            assert (
                windowed_data.shape[1] == self.stats.output_shape[0]
            ), f"Output shape: {windowed_data.shape}, Expected shape: {self.stats.output_shape}"
            assert (
                windowed_data.shape[2:] == self.stats.output_shape[1:]
            ), f"Output shape: {windowed_data.shape}, Expected shape: {self.stats.output_shape}"

    @staticmethod
    def _shape_numel(shape):
        return int(np.prod(shape)) if len(shape) > 0 else 1

    @staticmethod
    def _rest_numel(shape):
        return int(np.prod(shape[1:])) if len(shape) > 1 else 1


@register_context_operator(
    [ModalityType.TIMESERIES, ModalityType.AUDIO, ModalityType.EMBEDDING]
)
class WindowAggregation(Window):
    def __init__(
        self, aggregation_function="mean", window_size=10, pad=True, params=None
    ):
        if params is not None:
            aggregation_function = params["aggregation_function"]
            try:
                aggregation_function = aggregation_function()
            except:
                pass
            window_size = params["window_size"]
            pad = True
        super().__init__("WindowAggregation", aggregation_function)
        self.parameters["window_size"] = [5, 10, 15, 25, 50, 100]
        self.window_size = int(window_size)
        self.pad = pad

    def get_output_stats(self, input_stats: RepresentationStats) -> tuple:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 1:
            self.stats = RepresentationStats(
                input_stats.num_instances,
                (math.ceil(in_shape[0] / self.window_size),),
            )
        if len(in_shape) >= 2:
            self.stats = RepresentationStats(
                input_stats.num_instances,
                (math.ceil(in_shape[0] / self.window_size), *in_shape[1:]),
            )
        self.stats.output_shape_is_known = input_stats.output_shape_is_known
        return self.stats

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return 0

        out_seq_len = math.ceil(in_shape[0] / self.window_size)
        output_bytes = out_seq_len * self._rest_numel(in_shape)
        return (
            input_stats.num_instances * output_bytes * np.dtype(self.data_type).itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return {"cpu_peak_bytes": 0, "gpu_peak_bytes": 0}

        effective_seq_len = in_shape[0]
        in_numel = effective_seq_len * self._rest_numel(in_shape)
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        one_instance_bytes = in_numel * np.dtype(self.data_type).itemsize
        cpu_peak = (
            output_bytes * 2
            + one_instance_bytes * input_stats.num_instances
            + one_instance_bytes * self.window_size
            + 8 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}

    def execute(self, modality):
        windowed_data = []
        original_lengths = []
        for instance in modality.data:
            new_length = math.ceil(len(instance) / self.window_size)
            if modality.get_data_layout() == DataLayout.SINGLE_LEVEL:
                instance = np.array(instance)
                instance.setflags(write=False)
                windowed_instance = self.window_aggregate_single_level(
                    instance, new_length
                )
            else:
                instance = np.array(instance)
                instance.setflags(write=False)
                windowed_instance = self.window_aggregate_nested_level(
                    instance, new_length
                )
            original_lengths.append(new_length)
            windowed_data.append(windowed_instance)

        if self.pad and not isinstance(windowed_data, np.ndarray):
            target_length = max(original_lengths)
            sample_shape = windowed_data[0].shape

            padded_features = []
            for i, features in enumerate(windowed_data):
                current_len = original_lengths[i]

                if current_len < target_length:
                    padding_needed = target_length - current_len

                    pad_shape = (padding_needed,) + features.shape[1:]
                    padding = np.zeros(pad_shape)
                    padded = np.concatenate([features, padding], axis=0)

                    padded_features.append(padded)
                else:
                    padded_features.append(features)

            attention_masks = np.zeros((len(windowed_data), target_length))
            for i, length in enumerate(original_lengths):
                actual_length = min(length, target_length)
                attention_masks[i, :actual_length] = 1

            ModalityType(modality.modality_type).add_field_for_instances(
                modality.metadata, "attention_masks", attention_masks
            )

            windowed_data = np.array(padded_features)
            data_type = list(modality.metadata.values())[0]["data_layout"]["type"]
            if data_type != "str":
                windowed_data = windowed_data.astype(data_type)

        self.assert_output_stats(windowed_data)
        return windowed_data

    def window_aggregate_single_level(self, instance, new_length):
        if isinstance(instance, str):
            return instance

        result = []
        for i in range(0, new_length):
            result.append(
                self.aggregation_function.compute_feature(
                    instance[
                        i * self.window_size : i * self.window_size + self.window_size
                    ]
                )
            )

        return np.array(result)

    def window_aggregate_nested_level(self, instance, new_length):
        result = [[] for _ in range(0, new_length)]
        for i in range(0, new_length):
            result[i] = self.aggregation_function.compute_feature(
                instance[i * self.window_size : i * self.window_size + self.window_size]
            )

        return np.array(result)


@register_context_operator(
    [ModalityType.TIMESERIES, ModalityType.AUDIO, ModalityType.EMBEDDING]
)
class StaticWindow(Window):
    # TODO
    def __init__(self, aggregation_function="mean", num_windows=100, params=None):
        super().__init__("StaticWindow", aggregation_function)
        self.parameters["num_windows"] = [10, num_windows]
        self.num_windows = int(num_windows)

    def get_output_stats(self, input_stats: RepresentationStats) -> tuple:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) <= 1:
            self.stats = RepresentationStats(
                input_stats.num_instances, (self.num_windows,)
            )
        else:
            self.stats = RepresentationStats(
                input_stats.num_instances, (self.num_windows, *in_shape[1:])
            )
        self.stats.output_shape_is_known = input_stats.output_shape_is_known

        return self.stats

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return 0

        out_seq_len = self.num_windows
        output_bytes = out_seq_len * self._rest_numel(in_shape)
        return (
            input_stats.num_instances * output_bytes * np.dtype(self.data_type).itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return {"cpu_peak_bytes": 0, "gpu_peak_bytes": 0}
        effective_seq_len = in_shape[0]
        in_numel = effective_seq_len * self._rest_numel(in_shape)
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        one_instance_bytes = in_numel * np.dtype(self.data_type).itemsize
        cpu_peak = (
            output_bytes * 2
            + one_instance_bytes * input_stats.num_instances
            + 8 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}

    def execute(self, modality):
        windowed_data = []

        for instance in modality.data:
            window_size = len(instance) // self.num_windows
            remainder = len(instance) % self.num_windows
            output = []
            start = 0
            for i in range(0, self.num_windows):
                extra = 1 if i < remainder else 0
                end = start + window_size + extra
                window = instance[start:end]
                window.setflags(write=False)
                val = (
                    self.aggregation_function.compute_feature(window)
                    if len(window) > 0
                    else np.zeros_like(output[i - 1])
                )
                output.append(val)
                start = end

            windowed_data.append(output)
        windowed_data = np.array(windowed_data)
        self.assert_output_stats(windowed_data)
        return windowed_data


@register_context_operator(
    [ModalityType.TIMESERIES, ModalityType.AUDIO, ModalityType.EMBEDDING]
)
class DynamicWindow(Window):
    def __init__(self, aggregation_function="mean", num_windows=100, params=None):
        super().__init__("DynamicWindow", aggregation_function)
        self.parameters["num_windows"] = [10, num_windows]
        self.num_windows = int(num_windows)

    def get_output_stats(self, input_stats: RepresentationStats) -> tuple:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) <= 1:
            self.stats = RepresentationStats(
                input_stats.num_instances, (self.num_windows,)
            )
        else:
            self.stats = RepresentationStats(
                input_stats.num_instances, (self.num_windows, *in_shape[1:])
            )
        self.stats.output_shape_is_known = input_stats.output_shape_is_known
        return self.stats

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return 0

        out_seq_len = self.num_windows
        output_bytes = out_seq_len * self._rest_numel(in_shape)
        return (
            input_stats.num_instances * output_bytes * np.dtype(self.data_type).itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        in_shape = tuple(int(s) for s in input_stats.output_shape)
        if len(in_shape) == 0:
            return {"cpu_peak_bytes": 0, "gpu_peak_bytes": 0}
        effective_seq_len = in_shape[0]
        in_numel = effective_seq_len * self._rest_numel(in_shape)
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        one_instance_bytes = in_numel * np.dtype(self.data_type).itemsize
        cpu_peak = (
            output_bytes * 2
            + one_instance_bytes * input_stats.num_instances
            + 8 * 1024 * 1024
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}

    def execute(self, modality):
        windowed_data = []

        for instance in modality.data:
            N = len(instance)
            weights = np.geomspace(4, 256, num=self.num_windows)
            weights = weights / np.sum(weights)
            window_sizes = (weights * N).astype(int)
            window_sizes[-1] += N - np.sum(window_sizes)
            indices = np.cumsum(window_sizes)
            output = []
            start = 0
            for end in indices:
                window = instance[start:end]
                window.setflags(write=False)
                val = (
                    self.aggregation_function.compute_feature(window)
                    if len(window) > 0
                    else np.zeros_like(instance[0])
                )
                output.append(val)
                start = end

            windowed_data.append(output)
        windowed_data = np.array(windowed_data)
        self.assert_output_stats(windowed_data)
        return windowed_data
