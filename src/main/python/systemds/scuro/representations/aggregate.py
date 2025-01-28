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

from systemds.scuro.modality.modality import Modality


# TODO: make this a Representation and add a fusion method that fuses two modalities with each other


class Aggregation:
    def __init__(self, aggregation_function, field_name):
        self.aggregation_function = aggregation_function
        self.field_name = field_name

    def aggregate(self, modality):
        aggregated_modality = Modality(modality.modality_type, modality.metadata)
        aggregated_modality.data = []
        for i, instance in enumerate(modality.data):
            aggregated_modality.data.append([])
            for j, entry in enumerate(instance):
                if self.aggregation_function == "sum":
                    aggregated_modality.data[i].append(np.sum(entry, axis=0))
                elif self.aggregation_function == "mean":
                    aggregated_modality.data[i].append(np.mean(entry, axis=0))
                elif self.aggregation_function == "min":
                    aggregated_modality.data[i].append(np.min(entry, axis=0))
                elif self.aggregation_function == "max":
                    aggregated_modality.data[i].append(np.max(entry, axis=0))
                else:
                    raise ValueError("Invalid aggregation function")

        return aggregated_modality
