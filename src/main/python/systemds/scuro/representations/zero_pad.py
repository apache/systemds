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
from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.representation import derive_stats
from systemds.scuro.representations.unimodal import UnimodalRepresentation


class ZeroPad(UnimodalRepresentation):
    def __init__(self, width=None, input_width=None, params=None):
        super().__init__("ZeroPad", None)
        params = params or {}
        self.width = params.get("width", width)
        self.input_width = params.get("input_width", input_width)

    def transform(self, modality, aggregation=None):
        padded = TransformedModality(
            modality, self, self_contained=modality.self_contained
        )
        padded.data = modality.data
        Fusion._normalize_for_fusion(padded)
        padded.pad(max_len=self.width)
        return padded

    def get_output_stats(self, input_stats):
        shape = (*input_stats.output_shape[:-1], self.width)
        return derive_stats(input_stats, output_shape=shape)

    def get_current_parameters(self):
        return {"width": self.width, "input_width": self.input_width}
