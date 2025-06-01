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

from systemds.context import SystemDSContext
from systemds.operator.algorithm import gmm, gmmPredict


class TestGMM(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_lm_simple(self):
        a = self.sds.rand(500, 10, -100, 100, pdf="normal", seed=10)
        features = a  # training data all not outliers

        notOutliers = self.sds.rand(10, 10, -1, 1, seed=10)  # inside a
        outliers = self.sds.rand(10, 10, 1150, 1200, seed=10)  # outliers

        test = outliers.rbind(notOutliers)  # testing data half outliers

        n_gaussian = 4

        [_, _, _, _, mu, precision_cholesky, weight] = gmm(
            features, n_components=n_gaussian, seed=10
        )

        [_, pp] = gmmPredict(
            test, weight, mu, precision_cholesky, model=self.sds.scalar("VVV")
        )

        outliers = pp.max(axis=1) < 0.99
        ret = outliers.compute()

        self.assertTrue(ret.sum() == 10)


if __name__ == "__main__":
    unittest.main(exit=False)
