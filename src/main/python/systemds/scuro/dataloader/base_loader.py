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
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

class BaseLoader(ABC):
    def __init__(
        self, source_path: str, indices: List[str], chunk_size: Optional[int] = None
    ):
        """
        Base class to load raw data for a given list of indices and stores them in the data object
        :param source_path: The location where the raw data lies
        :param indices: A list of indices as strings that are corresponding to the file names
        :param chunk_size: An optional argument to load the data in chunks instead of all at once
        (otherwise please provide your own Dataloader that knows about the file name convention)
        """
        self.data = []
        self.metadata = (
            {}
        )  # TODO: check what the index should be for storing the metadata (file_name, counter, ...)
        self.source_path = source_path
        self.indices = indices
        self.chunk_size = chunk_size
        self.next_chunk = 0

        if self.chunk_size:
            self.num_chunks = int(len(self.indices) / self.chunk_size)

    def load(self):
        """
        Takes care of loading the raw data either chunk wise (if chunk size is defined) or all at once
        """
        if self.chunk_size:
            return self._load_next_chunk()

        return self._load(self.indices)

    def _load_next_chunk(self):
        """
        Loads the next chunk of data
        """
        self.data = []
        next_chunk_indices = self.indices[
            self.next_chunk * self.chunk_size : (self.next_chunk + 1) * self.chunk_size
        ]
        self.next_chunk += 1
        return self._load(next_chunk_indices)

    def _load(self, indices: List[str]):
        is_dir = True if os.path.isdir(self.source_path) else False

        if is_dir:
            _, ext = os.path.splitext(os.listdir(self.source_path)[0])
            for index in indices:
                self.extract(self.source_path + index + ext)
        else:
            self.extract(self.source_path, indices)

        return self.data

    @abstractmethod
    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        pass
    
    def create_timestamps(self, frequency, sample_length, start_datetime=None):
        start_time = start_datetime if start_datetime is not None else np.datetime64('1970-01-01T00:00:00.000000')
        time_increment = 1 / frequency
        time_increments_array = np.arange(sample_length) * np.timedelta64(int(time_increment * 1e6))
        timestamps = start_time + time_increments_array
        return timestamps
        
        
    def file_sanity_check(self, file):
        """
        Checks if the file can be found is not empty
        """
        try:
            file_size = os.path.getsize(file)
        except:
            raise (f"Error: File {0} not found!".format(file))

        if file_size == 0:
            raise ("File {0} is empty".format(file))
