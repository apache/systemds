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
from systemds.operator.algorithm import kmeans, kmeansPredict


class TestKMeans(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_500x2(self):
        """
        This test is based on statistics, that if we run kmeans, on a normal distributed dataset, centered around 0
        and use 4 clusters then they will be located in each one corner.
        """
        features = self.generate_matrices_for_k_means((500, 2), seed=1304)
        [res, classifications] = kmeans(features, k=4).compute()

        corners = set()
        for x in res:
            if x[0] > 0 and x[1] > 0:
                corners.add("pp")
            elif x[0] > 0 and x[1] < 0:
                corners.add("pn")
            elif x[0] < 0 and x[1] > 0:
                corners.add("np")
            else:
                corners.add("nn")
        self.assertTrue(len(corners) == 4)

    def test_500x2(self):
        """
        This test is based on statistics, that if we run kmeans, on a normal distributed dataset, centered around 0
        and use 4 clusters then they will be located in each one corner.
        This test uses the prediction builtin.
        """
        features = self.generate_matrices_for_k_means((500, 2), seed=1304)
        [c, _] = kmeans(features, k=4).compute()
        C = self.sds.from_numpy(c)
        elm = self.sds.from_numpy(np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]))
        res = kmeansPredict(elm, C).compute()
        corners = set()
        for x in res:
            if x == 1:
                corners.add("pp")
            elif x == 2:
                corners.add("pn")
            elif x == 3:
                corners.add("np")
            else:
                corners.add("nn")
        self.assertTrue(len(corners) == 4)

    def generate_matrices_for_k_means(self, dims, seed: int = 1234):
        np.random.seed(seed)
        mu, sigma = 0, 0.1
        s = np.random.normal(mu, sigma, dims[0] * dims[1])
        m1 = np.array(s, dtype=np.double)
        m1 = np.reshape(m1, (dims[0], dims[1]))

        return self.sds.from_numpy(m1)


if __name__ == "__main__":
    unittest.main(exit=False)
