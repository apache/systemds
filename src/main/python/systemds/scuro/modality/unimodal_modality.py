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
from systemds.scuro.modality.joined import JoinedModality
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType


class UnimodalModality(Modality):

    def __init__(self, data_loader: BaseLoader, modality_type: ModalityType):
        """
        This class represents a unimodal modality.
        :param data_loader: Defines how the raw data should be loaded
        :param modality_type: Type of the modality
        """
        super().__init__(modality_type, None)
        self.data_loader = data_loader
        
    def copy_from_instance(self):
        return type(self)(self.data_loader, self.modality_type)

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
            self.data_loader.chunk_size is not None
        )

        return joined_modality

    # TODO: add aggregation method like in join
    def apply_representation(self, representation, aggregation):
        new_modality = TransformedModality(self.modality_type, representation, self.data_loader.metadata)
        new_modality.data = []

        if self.data_loader.chunk_size:
            while (
                self.data_loader.next_chunk
                < self.data_loader.num_chunks
            ):
                self.extract_raw_data()
                new_modality.data.extend(representation.transform(self.data))
        else:
            if not self.data:
                self.extract_raw_data()
            new_modality.data = representation.transform(self.data)
            
        new_modality.update_metadata()
        return new_modality
