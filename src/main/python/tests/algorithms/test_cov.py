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
from systemds.operator.algorithm import cov


A = np.array([2, 4, 4, 2])
B = np.array([2, 4, 2, 4])
W = np.array([7, 1, 1, 1])
C = np.array([0, 1, 2])
D = np.array([2, 1, 0])
E = np.array([2, 1, 0])
F = np.array([2, 1, 0])


class TestCOV(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_cov1(self):
        sds_result = cov(self.sds.from_numpy(A), self.sds.from_numpy(B)).compute()
        np_result = np.cov(A, B)
        self.assertTrue(np.allclose(sds_result, np_result[0, 1], 1e-9))

    def test_cov2(self):
        sds_result = cov(self.sds.from_numpy(C), self.sds.from_numpy(D)).compute()
        np_result = np.cov(C, D)
        self.assertTrue(np.allclose(sds_result, np_result[0, 1], 1e-9))

    def test_cov3(self):
        sds_result = cov(self.sds.from_numpy(E), self.sds.from_numpy(F)).compute()
        np_result = np.cov(E, F)
        self.assertTrue(np.allclose(sds_result, np_result[0, 1], 1e-9))

    def test_cov4(self):
        sds_result = cov(
            self.sds.from_numpy(A), self.sds.from_numpy(B), self.sds.from_numpy(W)
        ).compute()
        np_result = np.cov(A, B, fweights=W)
        self.assertTrue(np.allclose(sds_result, np_result[0, 1], 1e-9))


if __name__ == "__main__":
    unittest.main(exit=False)
