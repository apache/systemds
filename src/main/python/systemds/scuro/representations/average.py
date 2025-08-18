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
import copy
from typing import List

import numpy as np

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.utils import pad_sequences

from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.drsearch.operator_registry import register_fusion_operator


@register_fusion_operator()
class Average(Fusion):
    def __init__(self):
        """
        Combines modalities using averaging
        """
        super().__init__("Average")
        self.needs_alignment = True
        self.associative = True
        self.commutative = True

    def execute(self, modalities: List[Modality]):
        data = copy.deepcopy(modalities[0].data)
        for i in range(1, len(modalities)):
            data += modalities[i].data

        data /= len(modalities)

        return np.array(data)
