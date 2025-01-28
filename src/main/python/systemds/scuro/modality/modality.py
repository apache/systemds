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

from systemds.scuro.modality.type import ModalityType


class Modality:

    def __init__(self, modalityType: ModalityType, metadata=None):
        """
        Parent class of the different Modalities (unimodal & multimodal)
        :param modality_type: Type of the modality
        """
        self.modality_type = modalityType
        self.schema = modalityType.get_schema()
        self.data = None
        self.data_type = None
        self.cost = None
        self.shape = None
        self.dataIndex = None
        self.metadata = metadata

    def get_modality_names(self) -> List[str]:
        """
        Extracts the individual unimodal modalities for a given transformed modality.
        """
        return [modality.name for modality in ModalityType if modality in self.modality_type]
    
    
    def update_metadata(self):
        md_copy = self.metadata
        self.metadata = {}
        for i, (md_k, md_v) in enumerate(md_copy.items()):
            updated_md = self.modality_type.update_metadata(md_v, self.data[i])
            self.metadata[md_k] = updated_md
            
            
    def window(self, windowSize, aggregationFunction, fieldName):
        pass
    