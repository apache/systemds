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

import shutil
import unittest

import numpy as np
from systemds.context import SystemDSContext


class TestWrite(unittest.TestCase):
    sds: SystemDSContext = None
    temp_dir: str = "tests/matrix/temp_write/"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_01(self):
        original = np.ones([10, 10])
        X = self.sds.from_numpy(original)
        X.write(self.temp_dir + "01").compute()

        NX = self.sds.read(self.temp_dir + "01")
        res = NX.compute()
        self.assertTrue(np.allclose(original, res))

    def test_write_02(self):
        original = np.array([[1, 2, 3, 4, 5]])
        X = self.sds.from_numpy(original)
        X.write(self.temp_dir + "02").compute()
        NX = self.sds.read(self.temp_dir + "02")
        res = NX.compute()
        self.assertTrue(np.allclose(original, res))


if __name__ == "__main__":
    unittest.main(exit=False)
