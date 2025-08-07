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


@register_context_operator()
class WindowAggregation(Context):
    def __init__(self, window_size=10, aggregation_function="mean", pad=True):
        parameters = {
            "window_size": [window_size],
            "aggregation_function": list(Aggregation().get_aggregation_functions()),
        }  # TODO: window_size should be dynamic and adapted to the shape of the data
        super().__init__("WindowAggregation", parameters)
        self.window_size = window_size
        self.aggregation_function = aggregation_function
        self.pad = pad

    @property
    def aggregation_function(self):
        return self._aggregation_function

    @aggregation_function.setter
    def aggregation_function(self, value):
        self._aggregation_function = Aggregation(value)

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
        instance = np.array(instance)
        num_cols = instance.shape[1] if instance.ndim > 1 else 1
        result = np.empty((new_length, num_cols))
        for i in range(0, new_length):
            result[i] = self.aggregation_function.aggregate_instance(
                instance[i * self.window_size : i * self.window_size + self.window_size]
            )

        if num_cols == 1:
            result = result.reshape(-1)
        return result

    def window_aggregate_nested_level(self, instance, new_length):
        result = [[] for _ in range(0, new_length)]
        data = np.stack(instance)
        for i in range(0, new_length):
            result[i] = self.aggregation_function.aggregate_instance(
                data[i * self.window_size : i * self.window_size + self.window_size]
            )

        return np.array(result)
