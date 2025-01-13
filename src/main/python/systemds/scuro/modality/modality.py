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

from systemds.scuro.modality.type import ModalityType


class Modality:

    def __init__(self, modality_type: ModalityType):
        """
        Parent class of the different Modalities (unimodal & multimodal)
        :param modality_type: Type of the modality
        """
        self.type = modality_type
        self.schema = modality_type.get_schema()
        self.data = None
        self.data_type = None
        self.cost = None
        self.shape = None
        self.data_index = None

    def get_modality_names(self) -> List[str]:
        """
        Extracts the individual unimodal modalities for a given transformed modality.
        """
        return [modality.name for modality in ModalityType if modality in self.type]
