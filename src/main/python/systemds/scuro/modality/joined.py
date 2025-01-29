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

from systemds.scuro.modality.joined_transformed import JoinedTransformedModality
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.aggregate import Aggregation


class JoinCondition:
    def __init__(self, leftField, rightField, joinType, alignment=None):
        self.leftField = leftField
        self.rightField = rightField
        self.join_type = joinType
        self.alignment = alignment


class JoinedModality(Modality):

    def __init__(
        self,
        modality_type,
        left_modality,
        right_modality,
        join_condition: JoinCondition,
        chunked_execution=False,
    ):
        """
        TODO
        :param modality_type: Type of the original modality(ies)
        """
        super().__init__(modality_type)
        self.aggregation = None
        self.joined_right = None
        self.left_modality = left_modality
        self.right_modality = right_modality
        self.condition = join_condition
        self.chunked_execution = chunked_execution # TODO: maybe move this into parent class
        self.left_type = type(left_modality)
        self.right_type = type(right_modality)
        if self.chunked_execution:
            self.chunk_left = left_modality.data_loader.chunk_size is not None

    def execute(self, right_starting_idx=0):
        self.joined_right = self.right_modality.copy_from_instance()

        for i, element in enumerate(self.left_modality.data):
            idx_1 = list(self.left_modality.metadata.values())[i + right_starting_idx][
                self.condition.leftField
            ]
            if (
                self.condition.alignment is None and self.condition.join_type == "<"
            ):  # TODO compute correct alignment timestamps/spatial params
                nextIdx = np.zeros(len(idx_1), dtype=int)
                nextIdx[:-1] = idx_1[1:]
                nextIdx[-1] = sys.maxsize

            idx_2 = list(self.right_modality.metadata.values())[i + right_starting_idx][
                self.condition.rightField
            ]
            self.joined_right.data.append([])

            c = 0
            # Assumes ordered lists (temporal)
            # TODO: need to extract the shape of the data from the metadata
            # video: list of lists of numpy array
            # audio: list of numpy array
            for j in range(0, len(idx_1)):
                self.joined_right.data[i].append([])
                other = np.array([])
                if self.condition.join_type == "<":
                    while c < len(idx_2) and idx_2[c] < nextIdx[j]:
                        if other.size == 0:
                            other = self.right_modality.data[i + right_starting_idx][c][np.newaxis, :]
                        else:
                            other = np.concatenate([other, self.right_modality.data[i + right_starting_idx][c][np.newaxis, :]], axis=0)
                        # other.append(self.right_modality.data[i][c])
                        c = c + 1
                else:
                    while c < len(idx_2) and idx_2[c] <= idx_1[j]:
                        if idx_2[c] == idx_1[j]:
                            other.append(self.right_modality.data[i + right_starting_idx][c])
                        c = c + 1
                        
                if len(other) == 0: # Audio and video length sometimes do not match so we add the average all audio samples for this specific frame
                    other = np.mean(self.right_modality.data[i + right_starting_idx], axis=0)[np.newaxis,:] # TODO: check correct loading for all data layouts, this is similar to missing data, add a different operation for htis
                self.joined_right.data[i][j] = other

    def apply_representation(self, representation, aggregation):
        self.aggregation = aggregation
        if self.chunked_execution:
            return self._handle_chunked_execution(representation)
        elif self.left_type.__name__.__contains__("Unimodal"):
            self.left_modality.extract_raw_data()
            if self.left_type == self.right_type:
                self.right_modality.extract_raw_data()
        elif self.right_type.__name__.__contains__("Unimodal"):
            self.right_modality.extract_raw_data()

        self.execute()
        left_transformed, right_transformed = self._apply_representation(representation)
        left_transformed.update_metadata()
        right_transformed.update_metadata()
        return JoinedTransformedModality(left_transformed, right_transformed, f'joined_{representation.name}')
        
        
    def aggregate(self, aggregation_function, field_name): # TODO: use the filed name to extract data entries from modalities
        self.aggregation = Aggregation(aggregation_function, field_name)
        
        if not self.chunked_execution and self.joined_right:
            return self.aggregation.aggregate(self.joined_right)
        
        return self
            

    def _handle_chunked_execution(self, representation):
        if self.left_type == self.right_type:
            return self._apply_representation_chunked(
                self.left_modality, self.right_modality, True, representation
            )
        elif self.chunk_left:
            return self._apply_representation_chunked(
                self.left_modality, self.right_modality, False, representation
            )
        else:
            return self._apply_representation_chunked(
                self.right_modality, self.left_modality, False, representation
            )

    def _apply_representation_chunked(
        self, chunk_modality, other_modality, chunk_other, representation
    ):
        new_left= TransformedModality(
            self.left_modality.modality_type,
            representation,
            {},
        )
        new_right = TransformedModality(
            self.right_modality.modality_type,
            representation,
            {},
        )
        while (
            chunk_modality.data_loader.next_chunk
            < chunk_modality.data_loader.num_chunks
        ):
            print(chunk_modality.data_loader.next_chunk
            )
            if chunk_other:
                other_modality.extract_raw_data()
                starting_idx = 0
            else:
                starting_idx = chunk_modality.data_loader.next_chunk * chunk_modality.data_loader.chunk_size
            chunk_modality.extract_raw_data()

            self.execute(starting_idx)
            
            left_transformed, right_transformed = self._apply_representation(representation)
            new_left.data.extend(
                left_transformed.data
            )
            new_left.metadata.update(left_transformed.metadata)
            new_right.data.extend(
                right_transformed.data
            )
            new_right.metadata.update(right_transformed.metadata)
        
        new_left.update_metadata()
        new_right.update_metadata()
        return JoinedTransformedModality(new_left, new_right, f'joined_{representation.name}')
    

    def _apply_representation(self, representation):
        left_transformed = representation.transform(self.left_modality)
        if self.aggregation:
            left_transformed = self.aggregation.window(left_transformed)
       
        right_transformed = representation.transform(self.joined_right)
        if self.aggregation:
            right_transformed = self.aggregation.window(right_transformed)
        
        return left_transformed, right_transformed