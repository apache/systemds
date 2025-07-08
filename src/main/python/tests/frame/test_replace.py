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

import json
import os
import shutil
import sys
import unittest

import numpy as np
import pandas as pd
from systemds.context import SystemDSContext


class TestReplaceFrame(unittest.TestCase):
    sds: SystemDSContext = None
    HOMES_PATH = "../../test/resources/datasets/homes/homes.csv"
    HOMES_SCHEMA = '"int,string,int,int,double,int,boolean,int,int"'
    JSPEC_PATH = "../../test/resources/datasets/homes/homes.tfspec_bin2.json"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def tearDown(self):
        pass

    def test_apply_recode_bin(self):
        F1 = self.sds.read(
            self.HOMES_PATH,
            data_type="frame",
            schema=self.HOMES_SCHEMA,
            format="csv",
            header=True,
        )
        ret = (
            F1.replace("north", "south")
            .replace("west", "south")
            .replace("east", "south")
            .compute()
        )
        self.assertTrue(any(ret.district == "south"))
        self.assertTrue(not (any(ret.district == "north")))


if __name__ == "__main__":
    unittest.main(exit=False)
