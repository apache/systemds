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


# TODO: move this into the aggregation class and add an aggregate() and a window(window_size) function there so they can use the same functionality.
class WindowAggregation:
    def __init__(self, window_size, aggregation_function):
        self.window_size = window_size
        self.aggregation_function = aggregation_function

    def window(self, modality):
        # data is a 2d array
        windowed_data = []
        for instance in modality.data:
            window_length = math.ceil(len(instance) / self.window_size)
            result = [[] for _ in range(0, window_length)]
            # if modality.schema["data_layout"]["representation"] == "list_of_lists_of_numpy_array":
            data = np.stack(instance)
            for i in range(0, window_length):
                result[i] = np.mean(
                    data[
                        i * self.window_size : i * self.window_size + self.window_size
                    ],
                    axis=0,
                )  # TODO: add actual aggregation function here

            windowed_data.append(result)

        return windowed_data
