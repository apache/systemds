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
s = 3.02


class TestBinaryOp(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_plus(self):
        self.assertTrue(
            np.allclose(
                (self.sds.from_numpy(m1) + self.sds.from_numpy(m2)).compute(), m1 + m2
            )
        )

    def test_minus(self):
        self.assertTrue(
            np.allclose(
                (self.sds.from_numpy(m1) - self.sds.from_numpy(m2)).compute(), m1 - m2
            )
        )

    def test_mul(self):
        self.assertTrue(
            np.allclose(
                (self.sds.from_numpy(m1) * self.sds.from_numpy(m2)).compute(), m1 * m2
            )
        )

    def test_div(self):
        self.assertTrue(
            np.allclose(
                (self.sds.from_numpy(m1) / self.sds.from_numpy(m2)).compute(), m1 / m2
            )
        )

    def test_plus3_rhs(self):
        self.assertTrue(np.allclose((self.sds.from_numpy(m1) + s).compute(), m1 + s))

    def test_plus3_lhs(self):
        self.assertTrue(np.allclose((s + self.sds.from_numpy(m1)).compute(), s + m1))

    def test_minus3_rhs(self):
        self.assertTrue(np.allclose((self.sds.from_numpy(m1) - s).compute(), m1 - s))

    def test_minus3_lhs(self):
        self.assertTrue(np.allclose((s - self.sds.from_numpy(m1)).compute(), s - m1))

    def test_mul3_rhs(self):
        self.assertTrue(np.allclose((self.sds.from_numpy(m1) * s).compute(), m1 * s))

    def test_mul3_lhs(self):
        self.assertTrue(np.allclose((s * self.sds.from_numpy(m1)).compute(), s * m1))

    def test_div3_rhs(self):
        self.assertTrue(np.allclose((self.sds.from_numpy(m1) / s).compute(), m1 / s))

    def test_div3_lhs(self):
        self.assertTrue(np.allclose((s / self.sds.from_numpy(m1)).compute(), s / m1))

    def test_matmul(self):
        self.assertTrue(
            np.allclose(
                (self.sds.from_numpy(m1) @ self.sds.from_numpy(m2)).compute(),
                m1.dot(m2),
            )
        )

    def test_matmul_chain(self):
        m3 = np.ones((m2.shape[1], 10), dtype=np.uint8)
        m = self.sds.from_numpy(m1) @ self.sds.from_numpy(m2) @ self.sds.from_numpy(m3)
        res = (m).compute()
        np_res = m1.dot(m2).dot(m3)
        self.assertTrue(np.allclose(res, np_res))

    def test_matmul_self(self):
        m = self.sds.from_numpy(m1).t() @ self.sds.from_numpy(m1)
        res = (m).compute()
        np_res = np.transpose(m1).dot(m1)
        self.assertTrue(np.allclose(res, np_res))

    def test_lt(self):
        self.assertTrue(
            np.allclose(
                (self.sds.from_numpy(m1) < self.sds.from_numpy(m2)).compute(), m1 < m2
            )
        )

    def test_gt(self):
        self.assertTrue(
            np.allclose(
                (self.sds.from_numpy(m1) > self.sds.from_numpy(m2)).compute(), m1 > m2
            )
        )

    def test_le(self):
        self.assertTrue(
            np.allclose(
                (self.sds.from_numpy(m1) <= self.sds.from_numpy(m2)).compute(), m1 <= m2
            )
        )

    def test_ge(self):
        self.assertTrue(
            np.allclose(
                (self.sds.from_numpy(m1) >= self.sds.from_numpy(m2)).compute(), m1 >= m2
            )
        )

    def test_abs(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).abs().compute(), np.abs(m1))
        )

    def test_lt3_rhs(self):
        self.assertTrue(np.allclose((self.sds.from_numpy(m1) < 3).compute(), m1 < 3))

    def test_lt3_lhs(self):
        self.assertTrue(np.allclose((3 < self.sds.from_numpy(m1)).compute(), 3 < m1))

    def test_gt3_rhs(self):
        self.assertTrue(np.allclose((3 > self.sds.from_numpy(m1)).compute(), 3 > m1))

    def test_le3_rhs(self):
        self.assertTrue(np.allclose((3 <= self.sds.from_numpy(m1)).compute(), 3 <= m1))

    def test_ge3_rhs(self):
        self.assertTrue(np.allclose((3 >= self.sds.from_numpy(m1)).compute(), 3 >= m1))


if __name__ == "__main__":
    unittest.main(exit=False)
