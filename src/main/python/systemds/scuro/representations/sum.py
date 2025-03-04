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
from systemds.scuro.representations.utils import pad_sequences

from systemds.scuro.representations.fusion import Fusion


class Sum(Fusion):
    def __init__(self):
        """
        Combines modalities using colum-wise sum
        """
        super().__init__("Sum")

    def transform(self, modalities: List[Modality]):
        max_emb_size = self.get_max_embedding_size(modalities)

        data = pad_sequences(modalities[0].data, maxlen=max_emb_size, dtype="float32")

        for m in range(1, len(modalities)):
            data += pad_sequences(
                modalities[m].data, maxlen=max_emb_size, dtype="float32"
            )

        return data
