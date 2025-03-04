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
        self.data = []
        self.data_type = None
        self.cost = None
        self.shape = None
        self.dataIndex = None
        self.metadata = metadata

    def get_modality_names(self) -> List[str]:
        """
        Extracts the individual unimodal modalities for a given transformed modality.
        """
        return [
            modality.name for modality in ModalityType if modality in self.modality_type
        ]

    def copy_from_instance(self):
        return type(self)(self.modality_type, self.metadata)

    def update_metadata(self):
        md_copy = self.metadata
        self.metadata = {}
        for i, (md_k, md_v) in enumerate(md_copy.items()):
            updated_md = self.modality_type.update_metadata(md_v, self.data[i])
            self.metadata[md_k] = updated_md

    def get_metadata_at_position(self, position: int):
        return self.metadata[self.dataIndex][position]

    def flatten(self):
        for num_instance, instance in enumerate(self.data):
            if type(instance) is np.ndarray:
                self.data[num_instance] = instance.flatten()
            elif type(instance) is list:
                self.data[num_instance] = np.array(
                    [item for sublist in instance for item in sublist]
                )

        self.data = np.array(self.data)
        return self

    def get_data_layout(self):
        if not self.data:
            return self.data

        if isinstance(self.data[0], list):
            return "list_of_lists_of_numpy_array"
        elif isinstance(self.data[0], np.ndarray):
            return "list_of_numpy_array"

    def get_data_shape(self):
        layout = self.get_data_layout()
        if not layout:
            return None

        if layout == "list_of_lists_of_numpy_array":
            return self.data[0][0].shape
        elif layout == "list_of_numpy_array":
            return self.data[0].shape

    def get_data_dtype(self):
        layout = self.get_data_layout()
        if not layout:
            return None

        if layout == "list_of_lists_of_numpy_array":
            return self.data[0][0].dtype
        elif layout == "list_of_numpy_array":
            return self.data[0].dtype

    def update_data_layout(self):
        if not self.data:
            return

        self.schema["data_layout"]["representation"] = self.get_data_layout()

        self.shape = self.get_data_shape()
        self.schema["data_layout"]["type"] = self.get_data_dtype()
