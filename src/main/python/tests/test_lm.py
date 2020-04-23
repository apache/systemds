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

import warnings
import unittest

import os
import sys

import numpy as np

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)

from systemds.context import SystemDSContext
from sklearn.linear_model import LinearRegression
import random

sds = SystemDSContext()

regressor = LinearRegression(fit_intercept=False)
shape = (random.randrange(1, 30), random.randrange(1, 30))
eps = 1e-05

class TestLm(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings(
            action="ignore", message="unclosed", category=ResourceWarning)

    def tearDown(self):
        warnings.filterwarnings(
            action="ignore", message="unclosed", category=ResourceWarning)

    def test_lm(self):
        X = np.random.rand(shape[0], shape[1])
        y = np.random.rand(shape[0], 1)

        try:
            sds_model_weights = sds.matrix(X).lm(sds.matrix(y)).compute()
            model = regressor.fit(X, y)

            model.coef_ = model.coef_.reshape(sds_model_weights.shape)
            self.assertTrue(np.allclose(sds_model_weights, model.coef_, eps))
        except Exception as e:
            self.assertTrue(False, "This should not raise an exception!")
            print(e)

    def test_lm_invalid_shape(self):
        X = np.random.rand(shape[0], 0)
        y = np.random.rand(0, 1)

        try:
            sds_model_weights = sds.matrix(X).lm(sds.matrix(y)).compute()
            self.assertTrue(False, "An exception was expected!")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    unittest.main(exit=False)
    sds.close()