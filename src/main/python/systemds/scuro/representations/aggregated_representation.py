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
from systemds.scuro.representations.representation import Representation
from systemds.scuro.representations.aggregate import Aggregation
import time


class AggregatedRepresentation(Representation):
    def __init__(self, aggregation="mean"):
        parameters = {
            "aggregation": list(Aggregation().get_aggregation_functions()),
        }
        super().__init__("AggregatedRepresentation", parameters)
        self.aggregation = Aggregation(aggregation)
        self.self_contained = True

    def transform(self, modality):
        start = time.perf_counter()
        aggregated_modality = TransformedModality(
            modality, self, self_contained=modality.self_contained
        )
        end = time.perf_counter()
        aggregated_modality.transform_time += end - start
        aggregated_modality.data = self.aggregation.execute(modality)
        return aggregated_modality
