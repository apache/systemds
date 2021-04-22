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
import shutil
import sys
import unittest

import pandas as pd
import numpy as np
import json
from systemds.context import SystemDSContext
from systemds.frame import Frame
from systemds.matrix import Matrix


class TestTransformEncode(unittest.TestCase):

    sds: SystemDSContext = None
    HOMES_PATH = "tests/frame/data/homes.csv"
    HOMES_SCHEMA = '"int,string,int,int,double,int,boolean,int,int"'

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def tearDown(self):
        pass

    def test_encode_recode(self):
        JSPEC_PATH = "tests/frame/data/homes.tfspec_recode2.json"
        with open(JSPEC_PATH) as jspec_file:
            JSPEC = json.load(jspec_file)
        F1 = self.sds.read(
            self.HOMES_PATH,
            data_type="frame",
            schema=self.HOMES_SCHEMA,
            format="csv",
            header=True,
        )
        pd_F1 = F1.compute()
        jspec = self.sds.read(JSPEC_PATH, data_type="scalar", value_type="string")
        X, M = F1.transform_encode(spec=jspec).compute()
        self.assertTrue(isinstance(X, np.ndarray))
        self.assertTrue(isinstance(M, pd.DataFrame))
        self.assertTrue(X.shape == pd_F1.shape)
        self.assertTrue(np.all(np.isreal(X)))
        for col_name in JSPEC["recode"]:
            self.assertTrue(M[col_name].nunique() == pd_F1[col_name].nunique())


if __name__ == "__main__":
    unittest.main(exit=False)
