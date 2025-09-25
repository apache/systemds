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

np.random.seed(7)
m1 = np.array(
    [
        [float("nan"), 2, 3, float("nan")],
        [5, float("nan"), 7, 8],
        [9, 10, float("nan"), 12],
        [float("nan"), 14, 15, float("nan")],
    ]
)

m2 = np.array(
    [
        [float("inf"), 2, 3, float("-inf")],
        [5, float("inf"), 7, 8],
        [9, 10, float("-inf"), 12],
        [float("inf"), 14, 15, float("-inf")],
    ]
)

dim = 100
m3 = np.random.random((dim * dim))
sel = np.random.randint(6, size=dim * dim)
m3[sel == 0] = float("nan")
m3[sel == 1] = float("inf")
m3[sel == 2] = float("-inf")
m3 = m3.reshape((dim, dim))


class TestIS_SPECIAL(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    def tearDown(self):
        self.sds.close()

    def test_na_basic(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.isNA().compute()
        np_result = np.isnan(m1)
        assert np.allclose(sds_result, np_result)

    def test_nan_basic(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.isNaN().compute()
        np_result = np.isnan(m1)
        assert np.allclose(sds_result, np_result)

    def test_inf_basic(self):
        sds_input = self.sds.from_numpy(m2)
        sds_result = sds_input.isInf().compute()
        np_result = np.isinf(m2)
        assert np.allclose(sds_result, np_result)

    def test_na_random(self):
        sds_input = self.sds.from_numpy(m3)
        sds_result = sds_input.isNA().compute()
        np_result = np.isnan(m3)
        assert np.allclose(sds_result, np_result)

    def test_nan_random(self):
        sds_input = self.sds.from_numpy(m3)
        sds_result = sds_input.isNaN().compute()
        np_result = np.isnan(m3)
        assert np.allclose(sds_result, np_result)

    def test_inf_random(self):
        sds_input = self.sds.from_numpy(m3)
        sds_result = sds_input.isInf().compute()
        np_result = np.isinf(m3)
        assert np.allclose(sds_result, np_result)

    def test_na_scalar1(self):
        self.assertTrue(self.sds.scalar(float("nan")).isNA() == 1)

    def test_na_scalar2(self):
        self.assertTrue(self.sds.scalar(1.0).isNA() == 0)

    def test_nan_scalar1(self):
        self.assertTrue(self.sds.scalar(float("nan")).isNaN() == 1)

    def test_nan_scalar2(self):
        self.assertTrue(self.sds.scalar(1.0).isNaN() == 0)

    def test_inf_scalar1(self):
        self.assertTrue(self.sds.scalar(float("nan")).isInf() == 0)

    def test_inf_scalar2(self):
        self.assertTrue(self.sds.scalar(1.0).isInf() == 0)

    def test_inf_scalar3(self):
        self.assertTrue(self.sds.scalar(float("inf")).isInf() == 1)

    def test_inf_scalar4(self):
        self.assertTrue(self.sds.scalar(float("-inf")).isInf() == 1)


if __name__ == "__main__":
    unittest.main()
