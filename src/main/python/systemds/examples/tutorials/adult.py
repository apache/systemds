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
import pandas as pd
import requests

class DataManager:

    _train_data_url: str
    _train_labels_url: str
    _test_data_url: str
    _test_labels_url: str

    _train_data_loc: str
    _train_labels_loc: str
    _test_data_loc: str
    _test_labels_loc: str

    _data_columns: []

    def __init__(self):
        self._train_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        self._test_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

        self._train_data_loc = "systemds/examples/tutorials/adult/train_data.csv"
        self._test_data_loc = "systemds/examples/tutorials/adult/test_data.csv"

        self._data_columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                   "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                   "income"]


    def get_train_data(self) -> np.array:
        self._get_data(self._train_data_url, self._train_data_loc)
        return self._parse_data(self._train_data_loc)\
            .drop(labels=self._data_columns[len(self._data_columns)-1], axis=1).to_numpy()

    def get_train_labels(self) -> np.array:
        self._get_data(self._train_data_url, self._train_data_loc)
        data_list = self._data_columns.copy()
        data_list.pop(len(self._data_columns)-1)
        return self._parse_data(self._train_data_loc).drop(labels=data_list, axis=1).to_numpy().flatten()

    def get_test_data(self) -> np.array:
        self._get_data(self._test_data_url, self._test_data_loc)
        return self._parse_data(self._test_data_loc)\
            .drop(labels=self._data_columns[len(self._data_columns)-1], axis=1).to_numpy()

    def get_test_labels(self) -> np.array:
        self._get_data(self._test_data_url, self._test_data_loc)
        data_list = self._data_columns.copy()
        data_list.pop(len(self._data_columns)-1)
        return self._parse_data(self._test_data_loc).drop(labels=data_list, axis=1).to_numpy().flatten()

    def _parse_data(self, loc) -> pd.DataFrame:
        return pd.read_csv(loc, header=None, names=self._data_columns)


    def _get_data(self, url, loc):
        if not os.path.isfile(loc):
            myfile = requests.get(url)
            folder = os.path.dirname(loc)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            with open(loc, 'wb') as f:
                f.write(myfile.content)
