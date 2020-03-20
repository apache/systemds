# -------------------------------------------------------------
#
# Modifications Copyright 2020 Graz University of Technology
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

# Make the `systemds` package importable
import os
import sys

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)

import unittest
from systemds.matrix import Matrix
import numpy as np

dim = 5
m1 = np.array(np.random.randint(100, size=dim * dim) + 1.01, dtype=np.double)
m1.shape = (dim, dim)
m2 = np.array(np.random.randint(5, size=dim * dim) + 1, dtype=np.double)
m2.shape = (dim, dim)
s = 3.02


class TestBinaryOp(unittest.TestCase):

    def test_plus(self):
        self.assertTrue(np.allclose((Matrix(m1) + Matrix(m2)).compute(), m1 + m2))

    def test_minus(self):
        self.assertTrue(np.allclose((Matrix(m1) - Matrix(m2)).compute(), m1 - m2))

    def test_mul(self):
        self.assertTrue(np.allclose((Matrix(m1) * Matrix(m2)).compute(), m1 * m2))

    def test_div(self):
        self.assertTrue(np.allclose((Matrix(m1) / Matrix(m2)).compute(), m1 / m2))

    # TODO arithmetic with numpy rhs

    # TODO arithmetic with numpy lhs

    def test_plus3(self):
        self.assertTrue(np.allclose((Matrix(m1) + s).compute(), m1 + s))

    def test_minus3(self):
        self.assertTrue(np.allclose((Matrix(m1) - s).compute(), m1 - s))

    def test_mul3(self):
        self.assertTrue(np.allclose((Matrix(m1) * s).compute(), m1 * s))

    def test_div3(self):
        self.assertTrue(np.allclose((Matrix(m1) / s).compute(), m1 / s))

    # TODO arithmetic with scala lhs

    def test_lt(self):
        self.assertTrue(np.allclose((Matrix(m1) < Matrix(m2)).compute(), m1 < m2))

    def test_gt(self):
        self.assertTrue(np.allclose((Matrix(m1) > Matrix(m2)).compute(), m1 > m2))

    def test_le(self):
        self.assertTrue(np.allclose((Matrix(m1) <= Matrix(m2)).compute(), m1 <= m2))

    def test_ge(self):
        self.assertTrue(np.allclose((Matrix(m1) >= Matrix(m2)).compute(), m1 >= m2))

    def test_abs(self):
        self.assertTrue(np.allclose(Matrix(m1).abs().compute(), np.abs(m1)))


if __name__ == "__main__":
    unittest.main()
