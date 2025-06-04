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
from copy import deepcopy
from typing import List

import numpy as np

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations import utils


class Modality:

    def __init__(self, modalityType: ModalityType, modality_id=-1, metadata={}, data_type=None):
        """
        Parent class of the different Modalities (unimodal & multimodal)
        :param modality_type: Type of the modality
        """
        self.modality_type = modalityType
        self.schema = modalityType.get_schema()
        self.metadata = metadata
        self.data = []
        self.data_type = data_type
        self.cost = None
        self.shape = None
        self.modality_id = modality_id

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        """
        This method ensures that the data layout in the metadata is updated when the data changes
        """
        self._data = value
        self.update_metadata()

    def get_modality_names(self) -> List[str]:
        """
        Extracts the individual unimodal modalities for a given transformed modality.
        """
        return [
            modality.name for modality in ModalityType if modality in self.modality_type
        ]

    def copy_from_instance(self):
        """
        Create a copy of the modality instance
        """
        return type(self)(self.modality_type, self.metadata)

    def update_metadata(self):
        """
        Updates the metadata of the modality (i.e.: updates timestamps)
        """
        if (
            not self.has_metadata()
            or not self.has_data()
            or len(self.data) < len(self.metadata)
        ):
            return

        md_copy = deepcopy(self.metadata)
        self.metadata = {}
        for i, (md_k, md_v) in enumerate(md_copy.items()):
            updated_md = self.modality_type.update_metadata(md_v, self.data[i])
            self.metadata[md_k] = updated_md

    def flatten(self, padding=True):
        """
        Flattens modality data by row-wise concatenation
        Prerequisite for some ML-models
        """
        max_len = 0
        for num_instance, instance in enumerate(self.data):
            if type(instance) is np.ndarray:
                self.data[num_instance] = instance.flatten()
            elif isinstance(instance, List):
                self.data[num_instance] = np.array(
                    [item for sublist in instance for item in sublist]
                ).flatten()
            max_len = max(max_len, len(self.data[num_instance]))

        if padding:
            for i, instance in enumerate(self.data):
                if isinstance(instance, np.ndarray):
                    if len(instance) < max_len:
                        padded_data = np.zeros(max_len, dtype=instance.dtype)
                        padded_data[: len(instance)] = instance
                        self.data[i] = padded_data
                else:
                    padded_data = []
                    for entry in instance:
                        padded_data.append(utils.pad_sequences(entry, max_len))
                    self.data[i] = padded_data
        self.data = np.array(self.data)
        return self

    def get_data_layout(self):
        if self.has_metadata():
            return list(self.metadata.values())[0]["data_layout"]["representation"]

        return None

    def has_data(self):
        return self.data is not None and len(self.data) != 0

    def has_metadata(self):
        return self.metadata is not None and self.metadata != {}
