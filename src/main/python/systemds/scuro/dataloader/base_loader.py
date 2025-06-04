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
import math

import numpy as np
from tensorflow.python.ops.numpy_ops.np_dtypes import int16


class BaseLoader(ABC):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        data_type: Union[np.dtype, str],
        chunk_size: Optional[int] = None,
        modality_type=None,
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
        self.modality_type = modality_type
        self._next_chunk = 0
        self._num_chunks = 1
        self._chunk_size = None
        self._data_type = data_type

        if chunk_size:
            self.chunk_size = chunk_size

    @property
    def chunk_size(self):
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        self._chunk_size = value
        self._num_chunks = int(math.ceil(len(self.indices) / self._chunk_size))

    @property
    def num_chunks(self):
        return self._num_chunks

    @property
    def next_chunk(self):
        return self._next_chunk

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, data_type):
        self._data_type = self.resolve_data_type(data_type)

    def reset(self):
        self._next_chunk = 0
        self.data = []
        self.metadata = {}

    def load(self):
        """
        Takes care of loading the raw data either chunk wise (if chunk size is defined) or all at once
        """
        if self._chunk_size:
            return self._load_next_chunk()

        return self._load(self.indices)

    def update_chunk_sizes(self, other):
        if not self._chunk_size and not other.chunk_size:
            return

        if (
            self._chunk_size
            and not other.chunk_size
            or self._chunk_size < other.chunk_size
        ):
            other.chunk_size = self.chunk_size
        else:
            self.chunk_size = other.chunk_size

    def _load_next_chunk(self):
        """
        Loads the next chunk of data
        """
        self.data = []
        next_chunk_indices = self.indices[
            self._next_chunk
            * self._chunk_size : (self._next_chunk + 1)
            * self._chunk_size
        ]
        self._next_chunk += 1
        return self._load(next_chunk_indices)

    def _load(self, indices: List[str]):
        file_names = self.get_file_names(indices)
        if isinstance(file_names, str):
            self.extract(file_names, indices)
        else:
            for file_name in file_names:
                self.extract(file_name)

        return self.data, self.metadata

    def get_file_names(self, indices=None):
        is_dir = True if os.path.isdir(self.source_path) else False
        file_names = []
        if is_dir:
            _, ext = os.path.splitext(os.listdir(self.source_path)[0])
            for index in self.indices if indices is None else indices:
                file_names.append(self.source_path + index + ext)
            return file_names
        else:
            return self.source_path

    @abstractmethod
    def extract(self, file: str, index: Optional[Union[str, List[str]]] = None):
        pass

    @staticmethod
    def file_sanity_check(file):
        """
        Checks if the file can be found is not empty
        """
        try:
            file_size = os.path.getsize(file)
        except:
            raise (f"Error: File {0} not found!".format(file))

        if file_size == 0:
            raise ("File {0} is empty".format(file))

    @staticmethod
    def resolve_data_type(data_type):
        if isinstance(data_type, str):
            if data_type.lower() in [
                "float16",
                "float32",
                "float64",
                "int16",
                "int32",
                "int64",
            ]:
                return np.dtype(data_type)
            else:
                raise ValueError(f"Unsupported data_type string: {data_type}")
        elif data_type in [
            np.float16,
            np.float32,
            np.float64,
            np.int16,
            np.int32,
            np.int64,
            str,
        ]:
            return data_type
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")
