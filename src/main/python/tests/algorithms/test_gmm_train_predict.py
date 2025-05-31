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

import shutil
import unittest

from systemds.context import SystemDSContext
from systemds.operator.algorithm import gmm, gmmPredict


class TestGMM(unittest.TestCase):

    model_dir: str = "tests/algorithms/readwrite/"
    model_path: str = model_dir + "model"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.model_dir)

    def test_train_and_predict(self):
        self.train()
        self.predict()

    def train(self):
        with SystemDSContext() as sds_train:
            a = sds_train.rand(500, 10, -100, 100, pdf="normal", seed=10)
            features = a  # training data all not outliers

            n_gaussian = 4

            [_, _, _, _, mu, precision_cholesky, weight] = gmm(
                features, nComponents=n_gaussian, seed=10
            )

            model = sds_train.list(mu, precision_cholesky, weight)
            model.write(self.model_path).compute()

    def predict(self):
        with SystemDSContext() as sds_predict:
            model = sds_predict.read(self.model_path)
            mu = model[1].as_matrix()
            precision_cholesky = model[2].as_matrix()
            weight = model[3].as_matrix()
            notOutliers = sds_predict.rand(10, 10, -1, 1, seed=10)  # inside a
            outliers = sds_predict.rand(10, 10, 1150, 1200, seed=10)  # outliers

            test = outliers.rbind(notOutliers)  # testing data half outliers

            [_, pp] = gmmPredict(
                test, weight, mu, precision_cholesky, model=sds_predict.scalar("VVV")
            )

            outliers = pp.max(axis=1) < 0.99
            ret = outliers.compute()

            self.assertTrue(ret.sum() == 10)


if __name__ == "__main__":
    unittest.main(exit=False)
