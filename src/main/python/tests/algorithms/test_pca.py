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
from systemds.operator.algorithm import pca


class TestPCA(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_500x2(self):
        """
        This test constructs a line of values in 2d space.
        That if fit correctly maps perfectly to 1d space.
        The check is simply if the input value was positive
        then the output value should be similar.
        """
        m1 = self.generate_matrices_for_pca(30, seed=1304)
        X = self.sds.from_numpy(m1)

        [res, model, _, _] = pca(X, K=1, scale="FALSE", center="FALSE").compute()
        for x, y in zip(m1, res):
            self.assertTrue((x[0] > 0 and y > 0) or (x[0] < 0 and y < 0))

    def test_simple(self):
        """
        line of numbers. Here the pca should return values that are double or close to double of the last value
        """
        m1 = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        [res, model, _, _] = pca(
            self.sds.from_numpy(m1), K=1, scale=False, center=False
        ).compute()
        for x in range(len(m1) - 1):
            self.assertTrue(abs(res[x + 1] - res[0] * (x + 2)) < 0.001)

    def generate_matrices_for_pca(self, dims: int, seed: int = 1234):
        np.random.seed(seed)

        mu, sigma = 0, 0.1
        s = np.random.normal(mu, sigma, dims)

        m1 = np.array(np.c_[np.copy(s) * 1, np.copy(s) * 0.3], dtype=np.double)

        return m1


if __name__ == "__main__":
    unittest.main(exit=False)
