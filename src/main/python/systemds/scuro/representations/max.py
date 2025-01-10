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
import itertools
from typing import List

import numpy as np

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.utils import pad_sequences

from systemds.scuro.representations.fusion import Fusion


class RowMax(Fusion):
    def __init__(self, split=4):
        """
        Combines modalities by computing the outer product of a modality combination and
        taking the row max
        """
        super().__init__("RowMax")
        self.split = split

    def transform(
        self,
        modalities: List[Modality],
    ):
        if len(modalities) < 2:
            return np.array(modalities[0].data)

        max_emb_size = self.get_max_embedding_size(modalities)

        padded_modalities = []
        for modality in modalities:
            d = pad_sequences(modality.data, maxlen=max_emb_size, dtype="float32")
            padded_modalities.append(d)

        split_rows = int(len(modalities[0].data) / self.split)

        data = []

        for combination in itertools.combinations(padded_modalities, 2):
            combined = None
            for i in range(0, self.split):
                start = split_rows * i
                end = (
                    split_rows * (i + 1)
                    if i < (self.split - 1)
                    else len(modalities[0].data)
                )
                m = np.einsum(
                    "bi,bo->bio", combination[0][start:end], combination[1][start:end]
                )
                m = m.max(axis=2)
                if combined is None:
                    combined = m
                else:
                    combined = np.concatenate((combined, m), axis=0)
            data.append(combined)

        data = np.stack(data)
        data = data.max(axis=0)

        return np.array(data)
