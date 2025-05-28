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

from systemds.scuro.representations import utils


class Aggregation:
    @staticmethod
    def _mean_agg(data):
        return np.mean(data, axis=0)

    @staticmethod
    def _max_agg(data):
        return np.max(data, axis=0)

    @staticmethod
    def _min_agg(data):
        return np.min(data, axis=0)

    @staticmethod
    def _sum_agg(data):
        return np.sum(data, axis=0)

    _aggregation_function = {
        "mean": _mean_agg.__func__,
        "max": _max_agg.__func__,
        "min": _min_agg.__func__,
        "sum": _sum_agg.__func__,
    }

    def __init__(self, aggregation_function="mean", pad_modality=False, params=None):
        if params is not None:
            aggregation_function = params["aggregation_function"]
            pad_modality = params["pad_modality"]

        if aggregation_function not in self._aggregation_function.keys():
            raise ValueError("Invalid aggregation function")

        self._aggregation_func = self._aggregation_function[aggregation_function]
        self.name = "Aggregation"
        self.pad_modality = pad_modality

        self.parameters = {
            "aggregation_function": aggregation_function,
            "pad_modality": pad_modality,
        }

    def execute(self, modality):
        data = []
        max_len = 0
        for i, instance in enumerate(modality.data):
            data.append([])
            if isinstance(instance, np.ndarray):
                aggregated_data = self._aggregation_func(instance)
            else:
                aggregated_data = []
                for entry in instance:
                    aggregated_data.append(self._aggregation_func(entry))
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

        return data

    def transform(self, modality):
        return self.execute(modality)

    def aggregate_instance(self, instance):
        return self._aggregation_func(instance)

    def get_aggregation_functions(self):
        return self._aggregation_function.keys()
