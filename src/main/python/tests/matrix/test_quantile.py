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
m = np.random.random_integers(9, size=100)
M = np.random.random_integers(9, size=300).reshape(100, 3)
p = np.array([0.25, 0.5, 0.75])
m2 = np.array([1, 2, 3, 4, 5])
w2 = np.array([1, 1, 1, 1, 5])


def weighted_quantiles(values, weights, quantiles=0.5):
    # DML quantile method corresponds to R type 7. Thus, the reference must interpolate the
    # weighted-rank pair (lo, hi) with g = h - floor(h) instead of picking a single rank.
    i = np.argsort(values)
    sv = values[i]
    c = np.cumsum(weights[i])
    n = float(c[-1])
    q = np.atleast_1d(quantiles).astype(float)
    h = (n - 1.0) * q + 1.0
    lo = np.clip(np.floor(h).astype(int), 1, int(n))
    hi = np.minimum(lo + 1, int(n))
    g = h - lo
    lo_val = sv[np.searchsorted(c, lo, side="left")]
    hi_val = sv[np.searchsorted(c, hi, side="left")]
    out = (1.0 - g) * lo_val + g * hi_val
    return out if np.ndim(quantiles) else float(out[0])


class TestQUANTILE(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    def tearDown(self):
        self.sds.close()

    def test_median_random1(self):
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.median().compute()
        np_result = np.median(m)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_median_random2(self):
        with self.assertRaises(RuntimeError):
            sds_input = self.sds.from_numpy(M)
            sds_input.median().compute()

    def test_weighted_median(self):
        sds_input = self.sds.from_numpy(m2)
        sds_input2 = self.sds.from_numpy(w2)
        sds_result = sds_input.median(sds_input2).compute()
        np_result = weighted_quantiles(m2, w2)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_quantile1(self):
        sds_p = self.sds.from_numpy(p)
        sds_result = self.sds.from_numpy(m).quantile(sds_p).compute()
        np_result = np.array(
            [weighted_quantiles(m, np.ones(m.shape), quantiles=q) for q in p]
        ).reshape(-1, 1)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_quantile2(self):
        sds_p = self.sds.from_numpy(p)
        sds_result = self.sds.from_numpy(m2).quantile(sds_p).compute()
        np_result = np.array(
            [weighted_quantiles(m2, np.ones(m.shape), quantiles=q) for q in p]
        ).reshape(-1, 1)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_quantile3(self):
        sds_p = self.sds.from_numpy(p)
        sds_w = self.sds.from_numpy(w2)
        sds_result = self.sds.from_numpy(m2).quantile(sds_p, sds_w).compute()
        np_result = np.array(
            [weighted_quantiles(m2, w2, quantiles=q) for q in p]
        ).reshape(-1, 1)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_quantile4(self):
        sds_w = self.sds.from_numpy(w2)
        quant = 0.3
        sds_result = self.sds.from_numpy(m2).quantile(quant, sds_w).compute()
        np_result = weighted_quantiles(m2, w2, quantiles=quant)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_quantile5(self):
        sds_w = self.sds.from_numpy(w2)
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m2).quantile("0.5", sds_w)

    def test_quantile6(self):
        sds_w = self.sds.from_numpy(w2)
        quant = 1.3
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m2).quantile(quant, sds_w)


if __name__ == "__main__":
    unittest.main()
