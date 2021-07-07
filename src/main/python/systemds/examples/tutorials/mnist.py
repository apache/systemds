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

import array
import functools
import gzip
import operator
import os
import struct

import numpy as np
import requests

__all__ = ["DataManager"]


class DataManager:

    _train_data_url: str
    _train_labels_url: str
    _test_data_url: str
    _test_labels_url: str

    _train_data_loc: str
    _train_labels_loc: str
    _test_data_loc: str
    _test_labels_loc: str

    def __init__(self):
        self._train_data_url = "https://systemds.apache.org/assets/datasets/mnist/train-images-idx3-ubyte.gz"
        self._train_labels_url = "https://systemds.apache.org/assets/datasets/mnist/train-labels-idx1-ubyte.gz"
        self._test_data_url = "https://systemds.apache.org/assets/datasets/mnist/t10k-images-idx3-ubyte.gz"
        self._test_labels_url = "https://systemds.apache.org/assets/datasets/mnist/t10k-labels-idx1-ubyte.gz"

        self._train_data_loc = "systemds/examples/tutorials/mnist/train_data.gz"
        self._train_labels_loc = "systemds/examples/tutorials/mnist/train_labels.gz"
        self._test_data_loc = "systemds/examples/tutorials/mnist/test_data.gz"
        self._test_labels_loc = "systemds/examples/tutorials/mnist/test_labels.gz"

    def get_train_data(self) -> np.array:
        self._get_data(self._train_data_url, self._train_data_loc)
        return self._parse_data(self._train_data_loc)

    def get_train_labels(self) -> np.array:
        self._get_data(self._train_labels_url, self._train_labels_loc)
        return self._parse_data(self._train_labels_loc)

    def get_test_data(self) -> np.array:
        self._get_data(self._test_data_url, self._test_data_loc)
        return self._parse_data(self._test_data_loc)

    def get_test_labels(self) -> np.array:
        self._get_data(self._test_labels_url, self._test_labels_loc)
        return self._parse_data(self._test_labels_loc)

    def _parse_data(self, loc):
        f = gzip.open if os.path.splitext(loc)[1] == '.gz' else open
        with f(loc, 'rb') as fd:
            return self._parse(fd)

    def _parse(self, fd):
        DATA_TYPES = {0x08: 'B',  # unsigned byte
                      0x09: 'b',  # signed byte
                      0x0b: 'h',  # short (2 bytes)
                      0x0c: 'i',  # int (4 bytes)
                      0x0d: 'f',  # float (4 bytes)
                      0x0e: 'd'}  # double (8 bytes)

        header = fd.read(4)
        zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
        data_type = DATA_TYPES[data_type]
        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                        fd.read(4 * num_dimensions))

        data = array.array(data_type, fd.read())
        data.byteswap()  # looks like array.array reads data as little endian

        expected_items = functools.reduce(operator.mul, dimension_sizes)

        return np.array(data).reshape(dimension_sizes)

    def _get_data(self, url, loc):
        if not os.path.isfile(loc):
            myfile = requests.get(url)
            folder = os.path.dirname(loc)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            with open(loc, 'wb') as f:
                f.write(myfile.content)
