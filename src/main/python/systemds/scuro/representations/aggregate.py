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

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations import utils


class Aggregation:
    @staticmethod
    def _mean_agg(data, aggregate_dim=0):
        return np.mean(data, axis=aggregate_dim)

    @staticmethod
    def _max_agg(data, aggregate_dim=0):
        return np.max(data, axis=aggregate_dim)

    @staticmethod
    def _min_agg(data, aggregate_dim=0):
        return np.min(data, axis=aggregate_dim)

    @staticmethod
    def _sum_agg(data, aggregate_dim=0):
        return np.sum(data, axis=aggregate_dim)

    _aggregation_function = {
        "mean": _mean_agg.__func__,
        "max": _max_agg.__func__,
        "min": _min_agg.__func__,
        "sum": _sum_agg.__func__,
    }

    def __init__(self, aggregation_function="mean", pad_modality=True, params=None):
        if params is not None:
            aggregation_function = params["aggregation_function"]
            pad_modality = params["pad_modality"]

        if aggregation_function not in list(self._aggregation_function.keys()):
            raise ValueError("Invalid aggregation function")

        self._aggregation_func = self._aggregation_function[aggregation_function]
        self.name = "Aggregation"
        self.pad_modality = pad_modality
        self.aggregation_function_name = aggregation_function

        self.parameters = {
            "aggregation_function": self._aggregation_function.keys(),
        }

    def get_current_parameters(self):
        return {
            "aggregation_function": self.aggregation_function_name,
            "pad_modality": self.pad_modality,
        }

    def execute(self, modality, aggregate_dim=(0,)):
        data = []
        max_len = 0
        for i, instance in enumerate(modality.data):
            data.append([])
            if isinstance(instance, np.ndarray) or isinstance(instance, list):
                if (
                    modality.modality_type == ModalityType.IMAGE
                    or modality.modality_type == ModalityType.VIDEO
                ) and instance.ndim > 2:
                    aggregated_data = instance.flatten()
                elif (
                    isinstance(instance, np.ndarray)
                    and instance.ndim == 2
                    and instance.shape[1] == 1
                ):
                    aggregated_data = instance.flatten()
                else:
                    aggregated_data = self._aggregation_func(instance, aggregate_dim)
            else:
                aggregated_data = []
                for entry in instance:
                    aggregated_data.append(self._aggregation_func(entry, aggregate_dim))

            if isinstance(aggregated_data, np.generic):
                aggregated_data = np.array(
                    [aggregated_data], dtype=aggregated_data.dtype
                )
            elif isinstance(aggregated_data, np.ndarray) and aggregated_data.ndim == 0:
                aggregated_data = aggregated_data.reshape(1)
            elif not isinstance(aggregated_data, (list, np.ndarray)):
                aggregated_data = np.array([aggregated_data])

            max_len = max(max_len, len(aggregated_data))
            data[i] = aggregated_data

        if self.pad_modality:
            for i, instance in enumerate(data):
                if isinstance(instance, np.ndarray):
                    if len(instance) < max_len:
                        padded_data = np.zeros(max_len, dtype=instance.dtype)
                        padded_data[: len(instance)] = instance
                        data[i] = padded_data
                else:
                    padded_data = []
                    for entry in instance:
                        padded_data.append(utils.pad_sequences(entry, max_len))
                    data[i] = padded_data
            data = np.asarray(data)

        return data

    def transform(self, modality):
        return self.execute(modality)

    def compute_feature(self, instance):
        return self._aggregation_func(instance)

    def get_aggregation_functions(self):
        return list(self._aggregation_function.keys())
