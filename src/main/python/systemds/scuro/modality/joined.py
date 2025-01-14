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
import sys

import numpy as np

from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.utils.join_condition import JoinCondition


class JoinedModality(Modality):

    def __init__(self, modality_type, primary, other, join_condition: JoinCondition):
        """
        TODO
        :param modality_type: Type of the original modality(ies)
        """
        super().__init__(modality_type)
        self.primary_modality = primary
        self.other_modality = other
        self.condition = join_condition
        self.chunked_execution = False
        self._check_chunked_data_extraction()

    def execute(self):
        self.primary_modality.extract_raw_data()
        self.data = {"other": []}
        self.other_modality.extract_raw_data()

        for i, element in enumerate(self.primary_modality.data):
            idx_1 = list(self.primary_modality.data_loader.metadata.values())[i][
                self.condition.field_1
            ]
            if (
                self.condition.alignment is None and self.condition.join_type == "<"
            ):  # TODO compute correct alignment timestamps/spatial params
                next_idx = np.zeros(len(idx_1), dtype=int)
                next_idx[:-1] = idx_1[1:]
                next_idx[-1] = sys.maxsize

            idx_2 = list(self.other_modality.data_loader.metadata.values())[i][
                self.condition.field_2
            ]

            c = 0
            for j in range(0, len(idx_1)):
                other = []
                if self.condition.join_type == "<":
                    while c < len(idx_2) and idx_2[c] < next_idx[j]:
                        other.append(self.other_modality.data[i][c])
                        c = c + 1
                else:
                    while c < len(idx_2) and idx_2[c] <= idx_1[j]:
                        if idx_2[c] == idx_1[j]:
                            other.append(self.other_modality.data[i][c])
                        c = c + 1

                self.data["other"].append(other)

    def apply_representation(self, representation):
        if self.chunked_execution:
            new_modality = TransformedModality(
                self.primary_modality.type, representation
            )

            while (
                self.primary_modality.data_loader.get_next_chunk_number()
                < self.primary_modality.data_loader.get_num_total_chunks()
            ):
                self.execute()

    def _check_chunked_data_extraction(self):
        if self.primary_modality.data_loader.get_chunk_size():
            if not self.other_modality.data_loader.get_chunk_size():
                self.other_modality.data_loader.update_chunk_size(
                    self.primary_modality.data_loader.get_chunk_size()
                )
            elif (
                self.other_modality.data_loader.get_chunk_size()
                > self.primary_modality.data_loader.get_chunk_size()
            ):
                self.primary_modality.data_loader.update_chunk_size(
                    self.other_modality.data_loader.get_chunk_size()
                )
            self.chunked_execution = True
        elif self.other_modality.data_loader.get_chunk_size():
            self.primary_modality.data_loader.update_chunk_size(
                self.other_modality.data_loader.get_chunk_size()
            )
            self.chunked_execution = True
