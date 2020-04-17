#-------------------------------------------------------------
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
#-------------------------------------------------------------

# Make the `systemds` package importable
import os
import sys
import warnings
import unittest
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)
from systemds.context import SystemDSContext

dim = 5
m1 = np.array(np.random.randint(100, size=dim * dim) + 1.01, dtype=np.double)
m1.shape = (dim, dim)
m2 = np.array(np.random.randint(5, size=dim * dim) + 1, dtype=np.double)
m2.shape = (dim, dim)

sds = SystemDSContext()


class TestMatrixAggFn(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore",
                                message="unclosed",
                                category=ResourceWarning)

    def tearDown(self):
        warnings.filterwarnings(action="ignore",
                                message="unclosed",
                                category=ResourceWarning)

    def test_sum1(self):
        self.assertTrue(np.allclose(sds.matrix(m1).sum().compute(), m1.sum()))

    def test_sum2(self):
        self.assertTrue(np.allclose(sds.matrix(m1).sum(axis=0).compute(), m1.sum(axis=0)))

    def test_sum3(self):
        self.assertTrue(np.allclose(sds.matrix(m1).sum(axis=1).compute(), m1.sum(axis=1).reshape(dim, 1)))

    def test_mean1(self):
        self.assertTrue(np.allclose(sds.matrix(m1).mean().compute(), m1.mean()))

    def test_mean2(self):
        self.assertTrue(np.allclose(sds.matrix(m1).mean(axis=0).compute(), m1.mean(axis=0)))

    def test_mean3(self):
        self.assertTrue(np.allclose(sds.matrix(m1).mean(axis=1).compute(), m1.mean(axis=1).reshape(dim, 1)))

    def test_full(self):
        self.assertTrue(np.allclose(sds.full((2, 3), 10.1).compute(), np.full((2, 3), 10.1)))

    def test_seq(self):
        self.assertTrue(np.allclose(sds.seq(3).compute(), np.arange(4).reshape(4, 1)))

    def test_var1(self):
        self.assertTrue(np.allclose(sds.matrix(m1).var().compute(), m1.var(ddof=1)))

    def test_var2(self):
        self.assertTrue(np.allclose(sds.matrix(m1).var(axis=0).compute(), m1.var(axis=0, ddof=1)))

    def test_var3(self):
        self.assertTrue(np.allclose(sds.matrix(m1).var(axis=1).compute(), m1.var(axis=1, ddof=1).reshape(dim, 1)))

    def test_rand_basic(self):
        seed = 15
        shape = (20, 20)
        min_max = (0, 1)
        sparsity = 0.2

        m = sds.rand(rows=shape[0], cols=shape[1], pdf="uniform", min=min_max[0], max=min_max[1],
                     seed=seed, sparsity=sparsity).compute()

        self.assertTrue(m.shape == shape)
        self.assertTrue((m.min() >= min_max[0]) and (m.max() <= min_max[1]))

        # sparsity
        m_flat = m.flatten('F')
        count, bins, patches = plt.hist(m_flat)

        non_zero_value_percent = sum(count[1:]) * 100 / count[0]
        e = 0.05
        self.assertTrue((non_zero_value_percent >= (sparsity - e) * 100)
                        and (non_zero_value_percent <= (sparsity + e) * 100))
        self.assertTrue(sum(count) == (shape[0] * shape[1]))

    def test_rand_distribution(self):
        seed = 15
        shape = (20, 20)
        min_max = (0, 1)

        m = sds.rand(rows=shape[0], cols=shape[1], pdf="uniform", min=min_max[0], max=min_max[1],
                     seed=seed).compute()

        m_flat = m.flatten('F')

        dist = best_distribution(m_flat)
        self.assertTrue(dist == 'uniform')

        m1 = sds.rand(rows=shape[0], cols=shape[1], pdf="normal", min=min_max[0], max=min_max[1],
                     seed=seed).compute()

        m1_flat = m1.flatten('F')

        dist = best_distribution(m1_flat)
        self.assertTrue(dist == 'norm')


def best_distribution(data):
    distributions = ['norm', 'uniform']
    result = dict()

    for dist in distributions:
        param = getattr(st, dist).fit(data)

        D, p_value = st.kstest(data, dist, args=param)
        result[dist] = p_value

    best_dist = max(result, key=result.get)
    return best_dist


if __name__ == "__main__":
    unittest.main(exit=False)
    sds.close()
