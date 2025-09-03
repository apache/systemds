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

import numpy as np
import math

from systemds.scuro.modality.type import DataLayout, ModalityType

from systemds.scuro.drsearch.operator_registry import register_context_operator
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.context import Context


class Window(Context):
    def __init__(self, name, aggregation_function):
        parameters = {
            "aggregation_function": list(Aggregation().get_aggregation_functions()),
        }
        super().__init__(name, parameters)
        self.aggregation_function = aggregation_function

    @property
    def aggregation_function(self):
        return self._aggregation_function

    @aggregation_function.setter
    def aggregation_function(self, value):
        self._aggregation_function = Aggregation(value)


@register_context_operator()
class WindowAggregation(Window):
    def __init__(self, window_size=10, aggregation_function="mean", pad=False):
        super().__init__("WindowAggregation", aggregation_function)
        self.parameters["window_size"] = [window_size]
        self.window_size = window_size
        self.pad = pad

    def execute(self, modality):
        windowed_data = []
        original_lengths = []
        for instance in modality.data:
            new_length = math.ceil(len(instance) / self.window_size)
            if modality.get_data_layout() == DataLayout.SINGLE_LEVEL:
                windowed_instance = self.window_aggregate_single_level(
                    instance, new_length
                )
            else:
                windowed_instance = self.window_aggregate_nested_level(
                    instance, new_length
                )
            original_lengths.append(new_length)
            windowed_data.append(windowed_instance)

        if self.pad and not isinstance(windowed_data, np.ndarray):
            target_length = max(original_lengths)
            sample_shape = windowed_data[0].shape
            is_1d = len(sample_shape) == 1

            padded_features = []
            for i, features in enumerate(windowed_data):
                current_len = original_lengths[i]

                if current_len < target_length:
                    padding_needed = target_length - current_len

                    if is_1d:
                        padding = np.zeros(padding_needed)
                        padded = np.concatenate([features, padding])
                    else:
                        feature_dim = features.shape[-1]
                        padding = np.zeros((padding_needed, feature_dim))
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

        return windowed_data

    def window_aggregate_single_level(self, instance, new_length):
        if isinstance(instance, str):
            return instance
        instance = np.array(copy.deepcopy(instance))

        result = []
        for i in range(0, new_length):
            result.append(
                self.aggregation_function.aggregate_instance(
                    instance[
                        i * self.window_size : i * self.window_size + self.window_size
                    ]
                )
            )

        return np.array(result)

    def window_aggregate_nested_level(self, instance, new_length):
        result = [[] for _ in range(0, new_length)]
        data = np.stack(copy.deepcopy(instance))
        for i in range(0, new_length):
            result[i] = self.aggregation_function.aggregate_instance(
                data[i * self.window_size : i * self.window_size + self.window_size]
            )

        return np.array(result)


@register_context_operator()
class StaticWindow(Window):
    def __init__(self, num_windows=100, aggregation_function="mean"):
        super().__init__("StaticWindow", aggregation_function)
        self.parameters["num_windows"] = [num_windows]
        self.num_windows = num_windows

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
                window = copy.deepcopy(instance[start:end])
                val = (
                    self.aggregation_function.aggregate_instance(window)
                    if len(window) > 0
                    else np.zeros_like(output[i - 1])
                )
                output.append(val)
                start = end

            windowed_data.append(output)
        return np.array(windowed_data)


@register_context_operator()
class DynamicWindow(Window):
    def __init__(self, num_windows=100, aggregation_function="mean"):
        super().__init__("DynamicWindow", aggregation_function)
        self.parameters["num_windows"] = [num_windows]
        self.num_windows = num_windows

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
                window = copy.deepcopy(instance[start:end])
                val = (
                    self.aggregation_function.aggregate_instance(window)
                    if len(window) > 0
                    else np.zeros_like(instance[0])
                )
                output.append(val)
                start = end
            windowed_data.append(output)

        return np.array(windowed_data)
