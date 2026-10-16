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
import hashlib
import json
import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Callable, Iterator, List, Optional, Tuple, Union
import math
from numbers import Integral

import numpy as np


class LazyFileSequence(Sequence):
    """List-like file references decoded only when a sample is requested."""

    def __init__(self, file_names: List[str], decoder: Callable[[str], object]):
        self.file_names = tuple(file_names)
        self.decoder = decoder

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.decoder(self.file_names[index])

    def __len__(self):
        return len(self.file_names)

    def subset(self, indices):
        return type(self)([self.file_names[i] for i in indices], self.decoder)


class BaseLoader(ABC):
    def __init__(
        self,
        source_path: str,
        indices: List[str],
        data_type: Union[np.dtype, str],
        chunk_size: Optional[int] = None,
        modality_type=None,
        ext=None,
    ):
        """
        Base class to load raw data for a given list of indices and stores them in the data object
        :param source_path: The location where the raw data lies
        :param indices: A list of indices as strings that are corresponding to the file names
        :param chunk_size: An optional argument to load the data in chunks instead of all at once
        (otherwise please provide your own Dataloader that knows about the file name convention)
        """
        self.data = []
        self.metadata = []
        self.source_path = source_path
        self.indices = indices
        self.modality_type = modality_type
        self._next_chunk = 0
        self._num_chunks = 1
        self._chunk_size = None
        self._data_type = data_type
        self._ext = ext
        self.stats = None
        self.chunk_size = chunk_size

    def cached_stats(self, files, options, compute):
        if os.environ.get("SCURO_DISABLE_STATS_CACHE", "0") == "1":
            return compute()
        if isinstance(files, (str, os.PathLike)):
            files = [files]
        fingerprint = []
        for file in files:
            path = os.path.realpath(os.fspath(file))
            stat = os.stat(path)
            fingerprint.append((path, stat.st_size, stat.st_mtime_ns))
        key = json.dumps(
            {
                "schema": 2,
                "loader": f"{type(self).__module__}.{type(self).__qualname__}",
                "files": fingerprint,
                "options": options,
            },
            sort_keys=True,
            default=str,
        ).encode()
        digest = hashlib.sha256(key).hexdigest()
        cache_root = os.environ.get("SCURO_STATS_CACHE_DIR")
        if cache_root is None:
            cache_root = os.path.join(
                os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                "scuro",
                "loader_stats",
            )
        cache_path = os.path.join(cache_root, f"{digest}.pkl")
        try:
            with open(cache_path, "rb") as cache_file:
                return pickle.load(cache_file)
        except (FileNotFoundError, EOFError, OSError, pickle.PickleError):
            stats = compute()
        try:
            os.makedirs(cache_root, exist_ok=True)
            temporary_path = f"{cache_path}.{os.getpid()}.tmp"
            with open(temporary_path, "wb") as cache_file:
                pickle.dump(stats, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, cache_path)
        except OSError:
            pass
        return stats

    @property
    def chunk_size(self):
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        if value is None:
            self._chunk_size = None
            self._num_chunks = 1
        else:
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError("chunk_size must be None or a positive integer")
            self._chunk_size = int(value)
            self._num_chunks = int(math.ceil(len(self.indices) / self._chunk_size))

        stats = getattr(self, "stats", None)
        if stats is not None and hasattr(stats, "num_instances"):
            stats.num_instances = (
                len(self.indices)
                if self._chunk_size is None
                else min(len(self.indices), self._chunk_size)
            )
        if stats is not None and hasattr(stats, "num_total_instances"):
            stats.num_total_instances = len(self.indices)

    @property
    def is_chunked(self):
        return self._chunk_size is not None

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
        self.metadata = []

    def load(self):
        """
        Takes care of loading the raw data either chunk wise (if chunk size is defined) or all at once
        """
        if self.is_chunked:
            return self._load_next_chunk()

        return self._load(self.indices)

    def iter_loaded_chunks(
        self, reset: bool = True
    ) -> Iterator[Tuple[list, dict, List[str]]]:
        if reset:
            self.reset()

        if not self.is_chunked:
            data, metadata = self.load()
            yield data, metadata, self.indices
            return

        while self._next_chunk < self._num_chunks:
            chunk_start = self._next_chunk * self._chunk_size
            chunk_end = (self._next_chunk + 1) * self._chunk_size
            chunk_indices = self.indices[chunk_start:chunk_end]
            data, metadata = self._load_next_chunk()
            yield data, metadata, chunk_indices

    def update_chunk_sizes(self, other):
        sizes = [
            size for size in (self.chunk_size, other.chunk_size) if size is not None
        ]
        if not sizes:
            return
        shared_size = min(sizes)
        self.chunk_size = shared_size
        other.chunk_size = shared_size

    def _load_next_chunk(self):
        """
        Loads the next chunk of data
        """
        self.data = []
        self.metadata = []
        next_chunk_indices = self.indices[
            self._next_chunk
            * self._chunk_size : (self._next_chunk + 1)
            * self._chunk_size
        ]
        self._next_chunk += 1
        return self._load(next_chunk_indices)

    def _load(self, indices: List[str]):
        if self.data is not None and len(self.data) == len(indices):
            return self.data, self.metadata

        file_names = self.get_file_names(indices)
        if isinstance(file_names, str):
            self.extract(file_names, indices)
        else:
            for i, file_name in enumerate(file_names):
                self.extract(file_name, indices[i])

        return self.data, self.metadata

    def get_file_names(self, indices=None):
        is_dir = True if os.path.isdir(self.source_path) else False
        file_names = []
        if is_dir:
            if self._ext is None:
                _, self._ext = os.path.splitext(os.listdir(self.source_path)[0])
            for index in self.indices if indices is None else indices:
                file_names.append(os.path.join(self.source_path, index + self._ext))
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
            raise ValueError(f"Error: File {0} not found!".format(file))

        if file_size == 0:
            raise ValueError("File {0} is empty".format(file))

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
