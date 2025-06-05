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
from functools import reduce
from operator import or_


from systemds.scuro.dataloader.base_loader import BaseLoader
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.joined import JoinedModality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.modality_identifier import ModalityIdentifier


class UnimodalModality(Modality):

    def __init__(self, data_loader: BaseLoader):
        """
        This class represents an unimodal modality.
        :param data_loader: Defines how the raw data should be loaded
        :param modality_type: Type of the modality
        """
        super().__init__(
            data_loader.modality_type,
            ModalityIdentifier().new_id(),
            {},
            data_loader.data_type,
        )
        self.data_loader = data_loader

    def copy_from_instance(self):
        new_instance = type(self)(self.data_loader)
        if self.metadata:
            new_instance.metadata = self.metadata.copy()
        return new_instance

    def get_metadata_at_position(self, position: int):
        if self.data_loader.chunk_size:
            return self.metadata[
                self.data_loader.chunk_size * self.data_loader.next_chunk + position
            ]

        return self.metadata[self.dataIndex][position]

    def extract_raw_data(self):
        """
        Uses the data loader to read the raw data from a specified location
        and stores the data in the data location.
        """
        self.data, self.metadata = self.data_loader.load()

    def join(self, other, join_condition):
        if isinstance(other, UnimodalModality):
            self.data_loader.update_chunk_sizes(other.data_loader)

        joined_modality = JoinedModality(
            reduce(or_, [other.modality_type], self.modality_type),
            self,
            other,
            join_condition,
            self.data_loader.chunk_size is not None,
        )

        if self.data_loader.chunk_size is None:
            self.extract_raw_data()
            joined_modality.execute(0)
            joined_modality.joined_right.update_metadata()

        return joined_modality

    def context(self, context_operator):
        if not self.has_data():
            self.extract_raw_data()

        transformed_modality = TransformedModality(self, context_operator)

        transformed_modality.data = context_operator.execute(self)
        return transformed_modality

    def aggregate(self, aggregation_function):
        if self.data is None:
            raise Exception("Data is None")

    def apply_representations(self, representations):
        # TODO
        pass

    def apply_representation(self, representation):
        new_modality = TransformedModality(
            self,
            representation,
        )
        new_modality.data = []

        if self.data_loader.chunk_size:
            self.data_loader.reset()
            while self.data_loader.next_chunk < self.data_loader.num_chunks:
                self.extract_raw_data()
                transformed_chunk = representation.transform(self)
                new_modality.data.extend(transformed_chunk.data)
                new_modality.metadata.update(transformed_chunk.metadata)
        else:
            if not self.data:
                self.extract_raw_data()
            new_modality = representation.transform(self)

        new_modality.update_metadata()
        return new_modality
