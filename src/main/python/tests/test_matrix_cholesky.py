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
from systemds.matrix import Matrix

np.random.seed(7)
shape = np.random.randint(1, 100)
A = np.random.rand(shape, shape)
# set A = MM^T and A is a positive definite matrix
A = np.matmul(A, A.transpose())

m1 = -np.random.rand(shape, shape)
m2 = np.asarray([[4, 9], [1, 4]])
m3 = np.random.rand(shape, shape + 1)

class TestCholesky(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_basic1(self):
        L = Matrix(self.sds, A).cholesky().compute()
        self.assertTrue(np.allclose(L, np.linalg.cholesky(A)))

    def test_basic2(self):
        L = Matrix(self.sds, A).cholesky().compute()
        # L * L.H = A
        self.assertTrue(np.allclose(A, np.dot(L, L.T.conj())))

    def test_pos_def(self):
        with self.assertRaises(ValueError) as context:
            Matrix(self.sds, m1).cholesky(safe=True).compute()

    def test_symmetric_matrix(self):
        np.linalg.cholesky(m2)
        with self.assertRaises(ValueError) as context:
            Matrix(self.sds, m2).cholesky(safe=True).compute()

    def test_asymetric_dim(self):
        with self.assertRaises(ValueError) as context:
            Matrix(self.sds, m3).cholesky().compute()


if __name__ == "__main__":
    unittest.main(exit=False)
