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
import zipfile

import pandas as pd
import requests
from systemds.context import SystemDSContext


class DataManager:

    _data_zip_url: str

    _train_data_url: str
    _train_labels_url: str
    _test_data_url: str
    _test_labels_url: str

    _train_data_loc: str
    _train_labels_loc: str
    _test_data_loc: str
    _test_labels_loc: str

    _data_columns: []
    _data_string_labels: []

    def __init__(self):
        self._data_zip_url = "https://systemds.apache.org/assets/datasets/adult/data.zip"
        self._data_zip_loc = "systemds/examples/tutorials/adult/data.zip"

        self._train_data_loc = "systemds/examples/tutorials/adult/train_data.csv"
        self._test_data_loc = "systemds/examples/tutorials/adult/test_data.csv"
        self._jspec_loc = "systemds/examples/tutorials/adult/jspec.json"

    def get_train_data_pandas(self) -> pd.DataFrame:
        self._get_data(self._train_data_loc)
        return self._parse_data(self._train_data_loc)\
            .drop(labels=["income"], axis=1)

    def get_train_data(self, sds: SystemDSContext) -> 'Frame':
        self._get_data(self._train_data_loc)
        return sds.read(self._train_data_loc)[:,0:14]

    def get_train_labels_pandas(self) -> pd.DataFrame:
        self._get_data(self._train_data_loc)
        return self._parse_data(self._train_data_loc)["income"]

    def get_train_labels(self, sds: SystemDSContext) -> 'Frame':
        self._get_data(self._train_data_loc)
        return sds.read(self._train_data_loc)[:,14]

    def get_test_data_pandas(self) -> pd.DataFrame:
        self._get_data(self._test_data_loc)
        return self._parse_data(self._test_data_loc)\
            .drop(labels=["income"], axis=1)
    
    def get_test_data(self, sds: SystemDSContext) -> 'Frame':
        self._get_data(self._test_data_loc)
        return sds.read(self._test_data_loc)[:,0:14]

    def get_test_labels_pandas(self) -> pd.DataFrame:
        self._get_data(self._test_data_loc)
        return self._parse_data(self._test_data_loc)["income"]

    def get_test_labels(self, sds: SystemDSContext) -> 'Frame':
        self._get_data(self._test_data_loc)
        return sds.read(self._test_data_loc)[:,14]

    def get_jspec_string(self) -> str:
        self._get_data(self._jspec_loc)
        with open(self._jspec_loc, "r") as f:
            return f.read()
    
    def get_jspec(self, sds: SystemDSContext) -> 'Scalar':
        self._get_data(self._jspec_loc)
        return sds.read(self._jspec_loc, data_type="scalar", value_type="string")

    def _parse_data(self, loc) -> pd.DataFrame:
        return pd.read_csv(loc)

    def _get_data(self, loc):
        if not os.path.isfile(loc):
            folder = os.path.dirname(loc)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            if not os.path.isfile(self._data_zip_loc):
                myZip = requests.get(self._data_zip_url)
                with open(self._data_zip_loc, 'wb') as f:
                    f.write(myZip.content)
            with zipfile.ZipFile(self._data_zip_loc) as z:
                z.extractall(folder)
