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
from systemds.operator.algorithm import multiLogReg


class TestMultiLogReg(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_simple(self):
        """
        Test simple, if the log reg splits a dataset where everything over 1 is label 1 and under 1 is 0.
        """
        # Generate data
        mu, sigma = 1, 0.1
        X = np.reshape(np.random.normal(mu, sigma,  500), (2,250))
        # All over 1 is true
        f = lambda x: x[0] > 1
        labels = f(X)
        # Y labels as double
        Y = np.array(labels, dtype=np.double)
        # Transpose X to fit input format.
        X = X.transpose()

        # Call algorithm
        bias = multiLogReg(Matrix(self.sds,X),Matrix(self.sds,Y)).compute()
        
        # Calculate result.
        res = np.reshape(np.dot(X, bias[:len(X[0])]) + bias[len(X[0])], (250))

        f2 = lambda x: x > 0
        accuracy = np.sum(labels == f2(res)) / 250 * 100

        self.assertTrue(accuracy > 98)


if __name__ == "__main__":
    unittest.main(exit=False)
