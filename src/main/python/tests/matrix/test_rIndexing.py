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

import unittest

import numpy as np
from systemds.context import SystemDSContext


class Test_rIndexing(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_1(self):
        npA = np.arange(0, 100).reshape(10, 10)
        m1 = self.sds.from_numpy(npA)
        npres = npA[4]
        res = m1[4].compute()
        self.assertTrue(np.allclose(res, npres))

    def test_2(self):
        npA = np.arange(0, 100).reshape(10, 10)
        m1 = self.sds.from_numpy(npA)
        npres = npA[4:5]
        res = m1[4:5].compute()
        self.assertTrue(np.allclose(res, npres))

    def test_3(self):
        npA = np.arange(0, 100).reshape(10, 10)
        m1 = self.sds.from_numpy(npA)
        # Invalid to slice with a step
        with self.assertRaises(ValueError) as context:
            res = m1[4:7:2].compute()

    def test_4(self):
        npA = np.arange(0, 100).reshape(10, 10)
        m1 = self.sds.from_numpy(npA)
        npres = npA[:, 4]
        res = m1[:, 4].compute().flatten()
        self.assertTrue(np.allclose(res, npres))

    def test_5(self):
        npA = np.arange(0, 100).reshape(10, 10)
        m1 = self.sds.from_numpy(npA)
        npres = npA[:, 4:6]
        res = m1[:, 4:6].compute()
        self.assertTrue(np.allclose(res, npres))

    def test_6(self):
        npA = np.arange(0, 100).reshape(10, 10)
        m1 = self.sds.from_numpy(npA)
        npres = npA[1:2, 4:6]
        res = m1[1:2, 4:6].compute()
        self.assertTrue(np.allclose(res, npres))

    def test_7(self):
        npA = np.arange(0, 100).reshape(10, 10)
        m1 = self.sds.from_numpy(npA)
        npres = npA[1, 4:6]
        res = m1[1, 4:6].compute()
        self.assertTrue(np.allclose(res, npres))

    def test_8(self):
        npA = np.arange(0, 100).reshape(10, 10)
        m1 = self.sds.from_numpy(npA)
        with self.assertRaises(NotImplementedError) as context:
            res = m1[1:, 4:6].compute()

    def test_9(self):
        npA = np.arange(0, 100).reshape(10, 10)
        m1 = self.sds.from_numpy(npA)
        with self.assertRaises(NotImplementedError) as context:
            res = m1[:3, 4:6].compute()


if __name__ == "__main__":
    unittest.main(exit=False)
