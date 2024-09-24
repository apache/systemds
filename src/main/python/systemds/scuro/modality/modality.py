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

from systemds.scuro.representations.representation import Representation


class Modality:
    
    def __init__(self, representation: Representation, start_index: int = 0, modality_name='', train_indices=None):
        """
        Parent class of the different Modalities
        :param representation: Specifies how the data should be represented for a specific modality
        :param start_index: Defines the first index used for the alignment
        :param modality_name: Name of the modality
        :param train_indices: List of indices used for train-test split
        """
        self.representation = representation
        self.start_index = start_index
        self.name = modality_name
        self.data = None
        self.train_indices = train_indices
    
    def read_chunk(self):
        """
        Extracts a data chunk of the modality according to the window size specified in params
        """
        raise NotImplementedError
    
    def read_all(self, indices):
        """
        Implemented for every unique modality to read all samples from a specified format
        :param indices: List of indices to be read
        """
        pass
