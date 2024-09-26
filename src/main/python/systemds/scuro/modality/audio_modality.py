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
import os

from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.unimodal import UnimodalRepresentation


class AudioModality(Modality):
    def __init__(
        self,
        file_path: str,
        representation: UnimodalRepresentation,
        train_indices=None,
        start_index: int = 0,
    ):
        """
        Creates an audio modality
        :param file_path: path to file where the audio embeddings are stored
        :param representation: Unimodal representation that indicates how to extract the data from the file
        """
        super().__init__(representation, start_index, "Audio", train_indices)
        self.file_path = file_path

    def file_sanity_check(self):
        """
        Checks if the file can be found is not empty
        """
        try:
            file_size = os.path.getsize(self.file_path)
        except:
            raise (f"Error: File {0} not found!".format(self.file_path))

        if file_size == 0:
            raise ("File {0} is empty".format(self.file_path))

    def read_chunk(self):
        pass

    def read_all(self, indices=None):
        self.data = self.representation.parse_all(
            self.file_path, indices=indices
        )  # noqa
