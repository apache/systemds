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
from systemds.operator.algorithm import multiLogReg, multiLogRegPredict


class TestMultiLogReg(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_simple(self):
        """
        Test simple, if the log reg splits a dataset where everything over 1 is label 2 and under 1 is 1.
        With manual classification.
        """
        [X, labels, Y] = self.gen_data()

        # Call algorithm
        bias = multiLogReg(
            self.sds.from_numpy(X), self.sds.from_numpy(Y), verbose=False
        ).compute()

        # Calculate result.
        res = np.reshape(np.dot(X, bias[: len(X[0])]) + bias[len(X[0])], (250))

        def f2(x):
            return (x < 0) + 1

        accuracy = np.sum(labels == f2(res)) / 250 * 100

        self.assertTrue(accuracy > 98)

    def test_using_predict(self):
        """
        Test the algorithm using the predict function.
        With builtin classification
        """
        [X, labels, Y] = self.gen_data()
        # Call algorithm
        bias = multiLogReg(
            self.sds.from_numpy(X), self.sds.from_numpy(Y), verbose=False
        ).compute()

        [m, y_pred, acc] = multiLogRegPredict(
            self.sds.from_numpy(X),
            self.sds.from_numpy(bias),
            Y=self.sds.from_numpy(Y),
            verbose=False,
        ).compute()

        self.assertTrue(acc > 98)

    def gen_data(self):
        np.random.seed(13241)
        # Generate data
        mu, sigma = 1, 0.1
        X = np.reshape(np.random.normal(mu, sigma, 500), (2, 250))

        # All over 1 is true
        def f(x):
            return (x[0] > 1) + 1

        labels = f(X)
        # Y labels as double
        Y = np.array(labels, dtype=np.double)
        # Transpose X to fit input format.
        X = X.transpose()
        return X, labels, Y


if __name__ == "__main__":
    unittest.main(exit=False)
