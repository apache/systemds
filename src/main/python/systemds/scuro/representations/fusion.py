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

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.representation import Representation


class Fusion(Representation):
    def __init__(self, name):
        """
        Parent class for different multimodal fusion types
        :param name: Name of the fusion type
        """
        super().__init__(name)

    def transform(self, modalities: List[Modality]):
        """
        Implemented for every child class and creates a fused representation out of
        multiple modalities
        :param modalities: List of modalities used in the fusion
        :return: fused data
        """
        raise f"Not implemented for Fusion: {self.name}"

    def get_max_embedding_size(self, modalities: List[Modality]):
        """
        Computes the maximum embedding size from a given list of modalities
        :param modalities: List of modalities
        :return: maximum embedding size
        """
        max_size = modalities[0].data.shape[1]
        for idx in range(1, len(modalities)):
            curr_shape = modalities[idx].data.shape
            if len(modalities[idx - 1].data) != curr_shape[0]:
                raise f"Modality sizes don't match!"
            elif curr_shape[1] > max_size:
                max_size = curr_shape[1]

        return max_size
