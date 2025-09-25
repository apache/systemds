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
m3 = np.array(np.random.randint(10, size=dim * dim * 10) + 1, dtype=np.double)
m3.shape = (dim * 10, dim)


class TestMatrixAggFn(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_sum1(self):
        self.assertTrue(np.allclose(self.sds.from_numpy(m1).sum().compute(), m1.sum()))

    def test_sum2(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).sum(axis=0).compute(), m1.sum(axis=0))
        )

    def test_sum3(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).sum(axis=1).compute(),
                m1.sum(axis=1).reshape(dim, 1),
            )
        )

    def test_sum4(self):
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m1).sum(2)

    def test_prod1(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).prod().compute(), np.prod(m1))
        )

    def test_prod2(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).prod(0).compute(), np.prod(m1, 0))
        )

    def test_prod3(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).prod(axis=1).compute(),
                np.prod(m1, 1).reshape(dim, 1),
            )
        )

    def test_prod4(self):
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m1).prod(2)

    def test_mean1(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).mean().compute(), m1.mean())
        )

    def test_mean2(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).mean(axis=0).compute(), m1.mean(axis=0))
        )

    def test_mean3(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).mean(axis=1).compute(),
                m1.mean(axis=1).reshape(dim, 1),
            )
        )

    def test_mean4(self):
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m1).mean(2)

    def test_full(self):
        self.assertTrue(
            np.allclose(self.sds.full((2, 3), 10.1).compute(), np.full((2, 3), 10.1))
        )

    def test_seq(self):
        self.assertTrue(
            np.allclose(self.sds.seq(3).compute(), np.arange(4).reshape(4, 1))
        )

    def test_var1(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).var().compute(), m1.var(ddof=1))
        )

    def test_var2(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).var(axis=0).compute(), m1.var(axis=0, ddof=1)
            )
        )

    def test_var3(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).var(axis=1).compute(),
                m1.var(axis=1, ddof=1).reshape(dim, 1),
            )
        )

    def test_var4(self):
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m1).var(2)

    def test_min1(self):
        self.assertTrue(np.allclose(self.sds.from_numpy(m1).min().compute(), m1.min()))

    def test_min2(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).min(axis=0).compute(), m1.min(axis=0))
        )

    def test_min3(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).min(axis=1).compute(),
                m1.min(axis=1).reshape(dim, 1),
            )
        )

    def test_min4(self):
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m1).min(2)

    def test_max1(self):
        self.assertTrue(np.allclose(self.sds.from_numpy(m1).max().compute(), m1.max()))

    def test_max2(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).max(axis=0).compute(), m1.max(axis=0))
        )

    def test_max3(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).max(axis=1).compute(),
                m1.max(axis=1).reshape(dim, 1),
            )
        )

    def test_max4(self):
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m1).max(2)

    def test_trace1(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).trace().compute(), m1.trace())
        )

    def test_trace2(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m2).trace().compute(), m2.trace())
        )

    def test_countDistinctApprox1(self):
        distinct = 100
        m = np.round(np.random.random((1000, 1000)) * (distinct - 1))
        # allow and error of 1%
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m).countDistinctApprox().compute(),
                len(np.unique(m)),
                1,
            )
        )

    def test_countDistinctApprox2(self):
        distinct = 1000
        m = np.round(np.random.random((10000, 100)) * (distinct - 1))
        # allow and error of 1%
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m).countDistinctApprox(0).compute(),
                [len(np.unique(col)) * 100 for col in m.T],
                10,
            )
        )

    def test_countDistinctApprox3(self):
        distinct = 1000
        m = np.round(np.random.random((100, 10000)) * (distinct - 1))
        # allow and error of 1%
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m).countDistinctApprox(1).compute(),
                np.array([[len(np.unique(col))] for col in m]),
                10,
            )
        )

    def test_countDistinctApprox4(self):
        m = np.round(np.random.random((2, 2)))
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m).countDistinctApprox(2)

    def test_countDistinct1(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).countDistinct().compute(), len(np.unique(m1))
            )
        )

    def test_countDistinct2(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m2).countDistinct().compute(), len(np.unique(m2))
            )
        )

    def test_countDistinct3(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m3).countDistinct().compute(), len(np.unique(m3))
            )
        )

    def test_countDistinct4(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).countDistinct(0).compute(),
                [len(np.unique(col)) for col in m1.T],
            )
        )

    def test_countDistinct5(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m2).countDistinct(0).compute(),
                [len(np.unique(col)) for col in m2.T],
            )
        )

    def test_countDistinct6(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m3).countDistinct(0).compute(),
                [len(np.unique(col)) for col in m3.T],
            )
        )

    def test_countDistinct7(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).countDistinct(1).compute(),
                np.array([[len(np.unique(col))] for col in m1]),
            )
        )

    def test_countDistinct8(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m2).countDistinct(1).compute(),
                np.array([[len(np.unique(col))] for col in m2]),
            )
        )

    def test_countDistinct9(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m3).countDistinct(1).compute(),
                np.array([[len(np.unique(col))] for col in m3]),
            )
        )

    def test_countDistinct10(self):
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m3).countDistinct(2)

    def test_sd1(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m1).sd().compute(), np.std(m1, ddof=1), 1e-9
            )
        )

    def test_sd2(self):
        self.assertTrue(
            np.allclose(
                self.sds.from_numpy(m2).sd().compute(), np.std(m2, ddof=1), 1e-9
            )
        )


if __name__ == "__main__":
    unittest.main(exit=False)
