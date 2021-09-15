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

dim = 5
np.random.seed(7)
m1 = np.array(np.random.randint(100, size=dim * dim) + 1.01, dtype=np.double)
m1.shape = (dim, dim)
m2 = np.array(np.random.randint(5, size=dim * dim) + 1, dtype=np.double)
m2.shape = (dim, dim)


class TestMatrixAggFn(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_sum1(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).sum().compute(), m1.sum()))

    def test_sum2(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).sum(axis=0).compute(), m1.sum(axis=0)))

    def test_sum3(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).sum(axis=1).compute(), m1.sum(axis=1).reshape(dim, 1)))

    def test_mean1(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).mean().compute(), m1.mean()))

    def test_mean2(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).mean(axis=0).compute(), m1.mean(axis=0)))

    def test_mean3(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).mean(axis=1).compute(), m1.mean(axis=1).reshape(dim, 1)))

    def test_full(self):
        self.assertTrue(np.allclose(
            self.sds.full( (2, 3), 10.1).compute(), np.full((2, 3), 10.1)))

    def test_seq(self):
        self.assertTrue(np.allclose(
            self.sds.seq(3).compute(), np.arange(4).reshape(4, 1)))

    def test_var1(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).var().compute(), m1.var(ddof=1)))

    def test_var2(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).var(axis=0).compute(), m1.var(axis=0, ddof=1)))

    def test_var3(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).var(axis=1).compute(), m1.var(axis=1, ddof=1).reshape(dim, 1)))

    def test_min1(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).min().compute(), m1.min()))

    def test_min2(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).min(axis=0).compute(), m1.min(axis=0)))

    def test_min3(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).min(axis=1).compute(), m1.min(axis=1).reshape(dim, 1)))

    def test_max1(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).max().compute(), m1.max()))

    def test_max2(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).max(axis=0).compute(), m1.max(axis=0)))

    def test_max3(self):
        self.assertTrue(np.allclose(
            self.sds.from_numpy(m1).max(axis=1).compute(), m1.max(axis=1).reshape(dim, 1)))

if __name__ == "__main__":
    unittest.main(exit=False)
