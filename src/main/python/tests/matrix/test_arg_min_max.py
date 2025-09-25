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
m = np.array([[1, 2, 3], [6, 5, 4], [8, 7, 9]])
M = np.random.random_integers(9, size=300).reshape(100, 3)
p = np.array([0.25, 0.5, 0.75])
m2 = np.array([1, 2, 3, 4, 5])
w2 = np.array([1, 1, 1, 1, 5])


def weighted_quantiles(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]


class TestARGMINMAX(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    def tearDown(self):
        self.sds.close()

    def test_argmin_basic1(self):
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.argmin(0).compute()
        np_result = np.argmin(m, axis=0).reshape(-1, 1)
        assert np.allclose(sds_result - 1, np_result, 1e-9)

    def test_argmin_basic2(self):
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.argmin(1).compute()
        np_result = np.argmin(m, axis=1).reshape(-1, 1)
        assert np.allclose(sds_result - 1, np_result, 1e-9)

    def test_argmin_basic3(self):
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.argmin().compute()
        np_result = np.argmin(m)
        assert np.allclose(sds_result - 1, np_result, 1e-9)

    def test_argmin_basic4(self):
        sds_input = self.sds.from_numpy(m)
        with self.assertRaises(ValueError):
            sds_input.argmin(3)

    def test_argmax_basic1(self):
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.argmax(0).compute()
        np_result = np.argmax(m, axis=0).reshape(-1, 1)
        assert np.allclose(sds_result - 1, np_result, 1e-9)

    def test_argmax_basic2(self):
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.argmax(1).compute()
        np_result = np.argmax(m, axis=1).reshape(-1, 1)
        assert np.allclose(sds_result - 1, np_result, 1e-9)

    def test_argmax_basic3(self):
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.argmax().compute()
        np_result = np.argmax(m)
        assert np.allclose(sds_result - 1, np_result, 1e-9)

    def test_argmax_basic4(self):
        sds_input = self.sds.from_numpy(m)
        with self.assertRaises(ValueError):
            sds_input.argmax(3)


if __name__ == "__main__":
    unittest.main()
