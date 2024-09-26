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
import torch

from torch import nn
from typing import List

import numpy as np

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.fusion import Fusion


class LSTM(Fusion):
    def __init__(self, width=128, depth=1, dropout_rate=0.1):
        """
        Combines modalities using an LSTM
        """
        super().__init__("LSTM")
        self.depth = depth
        self.width = width
        self.dropout_rate = dropout_rate
        self.unimodal_embeddings = {}

    def fuse(self, modalities: List[Modality], train_indices=None):
        size = len(modalities[0].data)

        result = np.zeros((size, 0))

        for modality in modalities:
            if modality.name in self.unimodal_embeddings.keys():
                out = self.unimodal_embeddings.get(modality.name)
            else:
                out = self.run_lstm(modality.data)
                self.unimodal_embeddings[modality.name] = out

            result = np.concatenate([result, out], axis=-1)

        return result

    def run_lstm(self, data):
        d = data.astype(np.float32)
        dim = d.shape[-1]
        d = torch.from_numpy(d)
        dropout_layer = torch.nn.Dropout(self.dropout_rate)

        for x in range(0, self.depth):
            lstm_x = nn.LSTM(dim, self.width, batch_first=True, bidirectional=True)
            dim = 2 * self.width
            d = lstm_x(d)[0]

        out = dropout_layer(d)

        if d.ndim > 2:
            out = torch.flatten(out, 1)

        return out.detach().numpy()
