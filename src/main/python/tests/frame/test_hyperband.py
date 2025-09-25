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

import os
import shutil
import sys
import unittest

import numpy as np
import pandas as pd
from systemds.context import SystemDSContext
from systemds.operator.algorithm import hyperband


class TestHyperband(unittest.TestCase):

    sds: SystemDSContext = None
    np.random.seed(42)
    X_train = np.random.rand(50, 10)
    y_train = np.sum(X_train, axis=1, keepdims=True) + np.random.rand(50, 1)
    X_val = np.random.rand(50, 10)
    y_val = np.sum(X_val, axis=1, keepdims=True) + np.random.rand(50, 1)
    params = 'list("reg", "tol", "maxi")'
    min_max_params = [[0, 20], [0.0001, 0.1], [5, 10]]
    param_ranges = np.array(min_max_params)

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def tearDown(self):
        pass

    def test_hyperband(self):
        x_train = self.sds.from_numpy(self.X_train)
        y_train = self.sds.from_numpy(self.y_train)
        x_val = self.sds.from_numpy(self.X_val)
        y_val = self.sds.from_numpy(self.y_val)
        paramRanges = self.sds.from_numpy(self.param_ranges)
        params = self.params
        [best_weights_mat, opt_hyper_params_df] = hyperband(
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            params=params,
            paramRanges=paramRanges,
            verbose=False,
        ).compute()
        self.assertTrue(isinstance(best_weights_mat, np.ndarray))
        self.assertTrue(best_weights_mat.shape[0] == self.X_train.shape[1])
        self.assertTrue(best_weights_mat.shape[1] == self.y_train.shape[1])

        self.assertTrue(isinstance(opt_hyper_params_df, pd.DataFrame))
        self.assertTrue(opt_hyper_params_df.shape[1] == 1)
        for i, hyper_param in enumerate(opt_hyper_params_df.values.flatten().tolist()):
            self.assertTrue(
                self.min_max_params[i][0] <= hyper_param <= self.min_max_params[i][1]
            )


if __name__ == "__main__":
    unittest.main(exit=False)
