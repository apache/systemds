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

m1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
m2 = np.random.randint(10, size=(10, 10))
m3 = np.random.random((10, 10))


def comsumprod(m):
    s = 0
    out = []
    for i in m:
        s = i[0] + i[1] * s
        out.append(s)
    return np.array(out).reshape(-1, 1)


class TestCUMBASE(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    def tearDown(self):
        self.sds.close()

    def test_cumsum_basic(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.cumsum().compute()
        np_result = np.cumsum(m1, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_cumsum_random1(self):
        sds_input = self.sds.from_numpy(m2)
        sds_result = sds_input.cumsum().compute()
        np_result = np.cumsum(m2, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_cumsum_random2(self):
        sds_input = self.sds.from_numpy(m3)
        sds_result = sds_input.cumsum().compute()
        np_result = np.cumsum(m3, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_cumprod_basic(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.cumprod().compute()
        np_result = np.cumprod(m1, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_cumprod_random1(self):
        sds_input = self.sds.from_numpy(m2)
        sds_result = sds_input.cumprod().compute()
        np_result = np.cumprod(m2, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_cumprod_random2(self):
        sds_input = self.sds.from_numpy(m3)
        sds_result = sds_input.cumprod().compute()
        np_result = np.cumprod(m3, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_cumsumprod_basic(self):
        m = m1[:, :2]  # 2-col matrix
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.cumsumprod().compute()
        exp_result = comsumprod(m)
        self.assertTrue(np.allclose(sds_result, exp_result, 1e-9))

    def test_cumsumprod_random1(self):
        m = m2[:, :2]
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.cumsumprod().compute()
        exp_result = comsumprod(m)
        self.assertTrue(np.allclose(sds_result, exp_result, 1e-9))

    def test_cumsumprod_random2(self):
        m = m3[:, :2]
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.cumsumprod().compute()
        exp_result = comsumprod(m)
        self.assertTrue(np.allclose(sds_result, exp_result, 1e-9))

    def test_cummin_random1(self):
        sds_input = self.sds.from_numpy(m2)
        sds_result = sds_input.cummin().compute()
        np_result = np.minimum.accumulate(m2, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_cummin_random2(self):
        sds_input = self.sds.from_numpy(m3)
        sds_result = sds_input.cummin().compute()
        np_result = np.minimum.accumulate(m3, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_cummax_random1(self):
        sds_input = self.sds.from_numpy(m2)
        sds_result = sds_input.cummax().compute()
        np_result = np.maximum.accumulate(m2, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_cummax_random2(self):
        sds_input = self.sds.from_numpy(m3)
        sds_result = sds_input.cummax().compute()
        np_result = np.maximum.accumulate(m3, 0)
        assert np.allclose(sds_result, np_result, 1e-9)


if __name__ == "__main__":
    unittest.main()
