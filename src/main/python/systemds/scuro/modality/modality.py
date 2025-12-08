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
from numpy.f2py.auxfuncs import throw_error

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations import utils


class Modality:

    def __init__(
        self, modalityType: ModalityType, modality_id=-1, metadata={}, data_type=None
    ):
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
        self.transform_time = None

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
        modality_names = [
            modality.name for modality in ModalityType if modality in self.modality_type
        ]
        modality_names.append(str(self.modality_id))
        return modality_names

    def copy_from_instance(self):
        """
        Create a copy of the modality instance
        """
        return type(self)(
            self.modality_type, self.modality_id, self.metadata, self.data_type
        )

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

        for i, (md_k, md_v) in enumerate(self.metadata.items()):
            md_v = selective_copy_metadata(md_v)
            updated_md = self.modality_type.update_metadata(md_v, self.data[i])
            self.metadata[md_k] = updated_md
            if i == 0:
                self.data_type = updated_md["data_layout"]["type"]

    def flatten(self, padding=False):
        """
        Flattens modality data by row-wise concatenation
        Prerequisite for some ML-models
        """
        max_len = 0
        data = []
        for num_instance, instance in enumerate(self.data):
            if type(instance) is np.ndarray:
                d = instance.flatten()
                max_len = max(max_len, len(d))
                data.append(d)
            elif isinstance(instance, List):
                d = np.array(
                    [item for sublist in instance for item in sublist]
                ).flatten()
                max_len = max(max_len, len(d))
                data.append(d)

        if padding:
            for i, instance in enumerate(data):
                if isinstance(instance, np.ndarray):
                    if len(instance) < max_len:
                        padded_data = np.zeros(max_len, dtype=instance.dtype)
                        padded_data[: len(instance)] = instance
                        data[i] = padded_data
                else:
                    padded_data = []
                    for entry in instance:
                        padded_data.append(utils.pad_sequences(entry, max_len))
                    data[i] = padded_data
        self.data = np.array(data)
        return self

    def pad(self, value=0, max_len=None):
        try:
            if max_len is None:
                result = np.array(self.data)
            elif isinstance(self.data, np.ndarray) and self.data.shape[1] == max_len:
                result = self.data
            else:
                raise "Needs padding to max_len"
        except:
            maxlen = (
                max([len(seq) for seq in self.data]) if max_len is None else max_len
            )

            result = np.full((len(self.data), maxlen), value, dtype=self.data_type)

            for i, seq in enumerate(self.data):
                data = seq[:maxlen]
                result[i, : len(data)] = data

                if self.has_metadata():
                    attention_mask = np.zeros(result.shape[1], dtype=np.int8)
                    attention_mask[: len(seq[:maxlen])] = 1
                    md_key = list(self.metadata.keys())[i]
                    if "attention_mask" in self.metadata[md_key]:
                        self.metadata[md_key]["attention_mask"] = attention_mask
                    else:
                        self.metadata[md_key].update({"attention_mask": attention_mask})
        # TODO: this might need to be a new modality (otherwise we loose the original data)
        self.data = result

    def get_data_layout(self):
        if self.has_metadata():
            return list(self.metadata.values())[0]["data_layout"]["representation"]

        return None

    def has_data(self):
        return self.data is not None and len(self.data) != 0

    def has_metadata(self):
        return self.metadata is not None and self.metadata != {}

    def is_aligned(self, other_modality):
        aligned = True
        for i in range(len(self.data)):
            if (
                list(self.metadata.values())[i]["data_layout"]["shape"]
                != list(other_modality.metadata.values())[i]["data_layout"]["shape"]
            ):
                aligned = False
                break

        return aligned


def selective_copy_metadata(metadata):
    if isinstance(metadata, dict):
        new_md = {}
        for k, v in metadata.items():
            if k == "data_layout":
                new_md[k] = v.copy() if isinstance(v, dict) else v
            elif isinstance(v, np.ndarray):
                new_md[k] = v
            else:
                new_md[k] = selective_copy_metadata(v)
        return new_md
    elif isinstance(metadata, (list, tuple)):
        return type(metadata)(selective_copy_metadata(item) for item in metadata)
    else:
        return metadata
