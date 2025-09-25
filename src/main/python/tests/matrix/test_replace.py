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
import random
import shutil
import sys
import unittest

import numpy as np
import pandas as pd
from systemds.context import SystemDSContext

np.random.seed(7)
shape = (25, 25)


class TestReplaceMatrix(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def tearDown(self):
        pass

    def test_replace_01(self):
        m = (
            self.sds.rand(min=0, max=2, rows=shape[0], cols=shape[1], seed=14)
            .round()
            .replace(1, 2)
            .compute()
        )
        self.assertTrue(1 not in m)
        self.assertTrue(2 in m)
        self.assertTrue(0 in m)


if __name__ == "__main__":
    unittest.main(exit=False)
