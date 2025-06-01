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
import random

import numpy as np
from systemds.context import SystemDSContext
from systemds.operator.algorithm import split

# Seed the randomness.
np.random.seed(7)


class TestOrder(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_basic(self):
        m = self.make_matrix()

        o = self.sds.from_numpy(m).compute()
        s = m
        self.assertTrue(np.allclose(o, s))

    def test_split(self):
        X = self.make_matrix()
        Y = self.make_matrix(cols=2)

        [p1, p2, p3, p4] = split(
            self.sds.from_numpy(X), self.sds.from_numpy(Y)
        ).compute()
        exp1 = X[:2]
        exp2 = X[2:]
        exp3 = Y[:2]
        exp4 = Y[2:]
        self.assertTrue(np.allclose(p1, exp1))
        self.assertTrue(np.allclose(p2, exp2))
        self.assertTrue(np.allclose(p3, exp3))
        self.assertTrue(np.allclose(p4, exp4))

    def test_split_2(self):
        rows = 10
        X = self.make_matrix(rows=rows)
        Y = self.make_matrix(rows=rows, cols=2)

        [p1, p2, p3, p4] = split(
            self.sds.from_numpy(X), self.sds.from_numpy(Y)
        ).compute()
        exp1 = X[:7]
        exp2 = X[7:]
        exp3 = Y[:7]
        exp4 = Y[7:]
        self.assertTrue(np.allclose(p1, exp1))
        self.assertTrue(np.allclose(p2, exp2))
        self.assertTrue(np.allclose(p3, exp3))
        self.assertTrue(np.allclose(p4, exp4))

    def test_split_3(self):
        rows = 100
        X = self.make_matrix(rows=rows)
        Y = self.make_matrix(rows=rows, cols=2)

        [p1, p2, p3, p4] = split(
            self.sds.from_numpy(X), self.sds.from_numpy(Y)
        ).compute()
        exp1 = X[:70]
        exp2 = X[70:]
        exp3 = Y[:70]
        exp4 = Y[70:]
        self.assertTrue(np.allclose(p1, exp1))
        self.assertTrue(np.allclose(p2, exp2))
        self.assertTrue(np.allclose(p3, exp3))
        self.assertTrue(np.allclose(p4, exp4))

    def test_split_4(self):
        rows = 100
        X = self.make_matrix(rows=rows)
        Y = self.make_matrix(rows=rows, cols=2)

        [p1, p2, p3, p4] = split(
            self.sds.from_numpy(X), self.sds.from_numpy(Y), f=0.2
        ).compute()
        exp1 = X[:20]
        exp2 = X[20:]
        exp3 = Y[:20]
        exp4 = Y[20:]
        self.assertTrue(np.allclose(p1, exp1))
        self.assertTrue(np.allclose(p2, exp2))
        self.assertTrue(np.allclose(p3, exp3))
        self.assertTrue(np.allclose(p4, exp4))

    def make_matrix(self, rows=4, cols=4):
        return np.random.rand(rows, cols)


if __name__ == "__main__":
    unittest.main(exit=False)
