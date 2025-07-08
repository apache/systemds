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

import math
import random
import unittest

import numpy as np
import scipy.stats as st
from systemds.context import SystemDSContext

np.random.seed(7)
# TODO Remove the randomness of the test, such that
#      inputs for the random operation is predictable
shape = (random.randrange(1, 25), random.randrange(1, 25))
dist_shape = (10, 15)
min_max = (0, 1)
sparsity = random.uniform(0.0, 1.0)
seed = 123
distributions = ["norm", "uniform"]


class TestRand(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_rand_shape(self):
        m = self.sds.rand(rows=shape[0], cols=shape[1]).compute()
        self.assertTrue(m.shape == shape)

    def test_rand_min_max(self):
        m = self.sds.rand(
            rows=shape[0], cols=shape[1], min=min_max[0], max=min_max[1]
        ).compute()
        self.assertTrue((m.min() >= min_max[0]) and (m.max() <= min_max[1]))

    def test_rand_sparsity(self):
        m = self.sds.rand(
            rows=shape[0], cols=shape[1], sparsity=sparsity, seed=0
        ).compute()
        non_zero_value_percent = np.count_nonzero(m) * 100 / np.prod(m.shape)

        self.assertTrue(math.isclose(non_zero_value_percent, sparsity * 100, rel_tol=5))

    def test_rand_uniform_distribution(self):
        m = self.sds.rand(
            rows=dist_shape[0],
            cols=dist_shape[1],
            pdf="uniform",
            min=min_max[0],
            max=min_max[1],
            seed=0,
        ).compute()

        dist = find_best_fit_distribution(m.flatten("F"), distributions)
        self.assertTrue(dist == "uniform")

    def test_rand_normal_distribution(self):
        m = self.sds.rand(
            rows=dist_shape[0],
            cols=dist_shape[1],
            pdf="normal",
            min=min_max[0],
            max=min_max[1],
            seed=0,
        ).compute()

        dist = find_best_fit_distribution(m.flatten("F"), distributions)
        self.assertTrue(dist == "norm")

    def test_rand_zero_shape(self):
        m = self.sds.rand(rows=0, cols=0).compute()
        self.assertTrue(np.allclose(m, np.array([[]])))

    def test_rand_invalid_shape(self):
        with self.assertRaises(ValueError) as context:
            self.sds.rand(rows=1, cols=-10).compute()

    def test_rand_invalid_pdf(self):
        with self.assertRaises(ValueError) as context:
            self.sds.rand(rows=1, cols=10, pdf="norm").compute()


def find_best_fit_distribution(data, distribution_lst):
    """
    Finds and returns the distribution of the distributions list that fits the data the best.
    :param data: flat numpy array
    :param distribution_lst: distributions to check
    :return: best distribution that fits the data
    """
    result = dict()

    for dist in distribution_lst:
        param = getattr(st, dist).fit(data)

        D, p_value = st.kstest(data, dist, args=param)
        result[dist] = p_value

    best_dist = max(result, key=result.get)
    return best_dist


if __name__ == "__main__":
    unittest.main(exit=False)
