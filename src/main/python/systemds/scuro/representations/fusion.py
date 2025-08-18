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
from typing import List

import numpy as np
from systemds.scuro import AggregatedRepresentation, Aggregation

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.representation import Representation
from systemds.scuro.utils.schema_helpers import get_shape


class Fusion(Representation):
    def __init__(self, name, parameters=None):
        """
        Parent class for different multimodal fusion types
        :param name: Name of the fusion type
        """
        super().__init__(name, parameters)
        self.associative = False
        self.commutative = False
        self.needs_alignment = False

    def transform(self, modalities: List[Modality]):
        """
        Implemented for every child class and creates a fused representation out of
        multiple modalities
        :param modalities: List of modalities used in the fusion
        :return: fused data
        """
        mods = []
        for modality in modalities:
            agg_modality = None
            if get_shape(modality.metadata) > 1:
                agg_operator = AggregatedRepresentation(Aggregation())
                agg_modality = agg_operator.transform(modality)
            mods.append(agg_modality if agg_modality else modality)

        if self.needs_alignment:
            max_len = self.get_max_embedding_size(mods)
            for modality in mods:
                modality.pad(max_len=max_len)
        return self.execute(mods)

    def execute(self, modalities: List[Modality]):
        raise f"Not implemented for Fusion: {self.name}"

    def get_max_embedding_size(self, modalities: List[Modality]):
        """
        Computes the maximum embedding size from a given list of modalities
        :param modalities: List of modalities
        :return: maximum embedding size
        """
        if isinstance(modalities[0].data[0], list):
            max_size = modalities[0].data[0][0].shape[1]
        elif isinstance(modalities[0].data, np.ndarray):
            max_size = modalities[0].data.shape[1]
        else:
            max_size = modalities[0].data[0].shape[1]
        for idx in range(1, len(modalities)):
            if isinstance(modalities[idx].data[0], list):
                curr_shape = modalities[idx].data[0][0].shape
            elif isinstance(modalities[idx].data, np.ndarray):
                curr_shape = modalities[idx].data.shape
            else:
                curr_shape = modalities[idx].data[0].shape
            if len(modalities[idx - 1].data) != len(modalities[idx].data):
                raise f"Modality sizes don't match!"
            elif curr_shape[1] > max_size:
                max_size = curr_shape[1]

        return max_size
