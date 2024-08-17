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

from modality.modality import Modality
from keras.api.preprocessing.sequence import pad_sequences

from representations.fusion import Fusion


class Concatenation(Fusion):
    def __init__(self, padding=True):
        """
        Combines modalities using concatenation
        """
        super().__init__('Concatenation')
        self.padding = padding

    def fuse(self, modalities: List[Modality]):
        max_emb_size = self.get_max_embedding_size(modalities)
        
        size = len(modalities[0].data)
        data = np.zeros((size, 0))
        
        for modality in modalities:
            if self.padding:
                data = np.concatenate(pad_sequences(modality.data, maxlen=max_emb_size, dtype='float32', padding='post'), axis=1)
            else:
                data = np.concatenate([data, modality.data], axis=1)
      
        return self.scale_data(data, modalities[0].train_indices)
