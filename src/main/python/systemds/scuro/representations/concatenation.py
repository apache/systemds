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

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.utils import pad_sequences

from systemds.scuro.representations.fusion import Fusion


class Concatenation(Fusion):
    def __init__(self, padding=True):
        """
        Combines modalities using concatenation
        """
        super().__init__("Concatenation")
        self.padding = padding

    def transform(self, modalities: List[Modality]):
        if len(modalities) == 1:
            return np.array(modalities[0].data)

        max_emb_size = self.get_max_embedding_size(modalities)
        size = len(modalities[0].data)

        if modalities[0].data.ndim > 2:
            data = np.zeros((size, max_emb_size, 0))
        else:
            data = np.zeros((size, 0))

        for modality in modalities:
            if self.padding:
                data = np.concatenate(
                    [
                        data,
                        pad_sequences(
                            modality.data, maxlen=max_emb_size, dtype="float32"
                        ),
                    ],
                    axis=-1,
                )
            else:
                data = np.concatenate([data, modality.data], axis=-1)

        return np.array(data)
