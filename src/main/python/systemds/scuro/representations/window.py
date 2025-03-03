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

from systemds.scuro.modality.type import DataLayout


class WindowAggregation:
    def __init__(self, window_size, aggregation_function):
        self.window_size = window_size
        self.aggregation_function = aggregation_function

    def execute(self, modality):
        windowed_data = []
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

            windowed_data.append(windowed_instance)

        return windowed_data

    def window_aggregate_single_level(self, instance, new_length):
        num_cols = instance.shape[1] if instance.ndim > 1 else 1
        result = np.empty((new_length, num_cols))
        for i in range(0, new_length):
            result[i] = self.aggregation_function.aggregate_instance(
                instance[i * self.window_size : i * self.window_size + self.window_size]
            )

        return result

    def window_aggregate_nested_level(self, instance, new_length):
        result = [[] for _ in range(0, new_length)]
        data = np.stack(instance)
        for i in range(0, new_length):
            result[i] = self.aggregation_function.aggregate_instance(
                data[i * self.window_size : i * self.window_size + self.window_size]
            )

        return result
