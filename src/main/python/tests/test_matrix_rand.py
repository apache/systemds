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
import unittest
import numpy as np
import scipy.stats as st
import random

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)
from systemds.context import SystemDSContext


shape = (random.randrange(0, 50), random.randrange(0, 50))
min_max = (0, 1)
sparsity = 0.2
distributions = ['norm', 'uniform']

sds = SystemDSContext()


class TestRand(unittest.TestCase):
    def test_rand_shape(self):
        m = sds.rand(rows=shape[0], cols=shape[1]).compute()
        self.assertTrue(m.shape == shape)

    def test_rand_min_max(self):
        m = sds.rand(rows=shape[0], cols=shape[1], min=min_max[0], max=min_max[1]).compute()
        self.assertTrue((m.min() >= min_max[0]) and (m.max() <= min_max[1]))

    def test_rand_sparsity(self):
        m = sds.rand(rows=shape[0], cols=shape[1], sparsity=sparsity).compute()

        m_flat = m.flatten('F')
        count, bins = np.histogram(m_flat)

        non_zero_value_percent = sum(count[1:]) * 100 / count[0]
        e = 0.05
        self.assertTrue(sum(count) == (shape[0] * shape[1]) and (non_zero_value_percent >= (sparsity - e) * 100)
                        and (non_zero_value_percent <= (sparsity + e) * 100))

    def test_rand_uniform_distribution(self):
        m = sds.rand(rows=shape[0], cols=shape[1], pdf="uniform", min=min_max[0], max=min_max[1],).compute()

        dist = find_best_fit_distribution(m.flatten('F'), distributions)
        self.assertTrue(dist == 'uniform')

    def test_rand_normal_distribution(self):
        m = sds.rand(rows=shape[0], cols=shape[1], pdf="normal", min=min_max[0], max=min_max[1]).compute()

        dist = find_best_fit_distribution(m.flatten('F'), distributions)
        self.assertTrue(dist == 'norm')

    def test_rand_invalid_shape(self):
        try:
            sds.rand(rows=1, cols=-10).compute()
            self.assertTrue(False)
        except Exception as e:
            print(e)

    def test_rand_invalid_pdf(self):
        try:
            sds.rand(rows=1, cols=10, pdf="norm").compute()
            self.assertTrue(False)
        except Exception as e:
            print(e)


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
    sds.close()