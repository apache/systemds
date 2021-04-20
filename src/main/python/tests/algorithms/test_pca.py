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
from systemds.operator.algorithm import pca
from systemds.operator import OperationNode2
from systemds.script_building.dag import OutputType


class TestPCA(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_500x2(self):
        """
        This test constructs a line of values in 2d space. 
        That if fit correctly maps perfectly to 1d space.
        The check is simply if the input value was positive
        then the output value should be similar.
        """
        m1 = self.generate_matrices_for_pca(30, seed=1304)
        X = Matrix(self.sds, m1)
        # print(features)
        [res, model, _, _] = pca(X, K=1, scale="FALSE", center="FALSE").compute()
        for (x, y) in zip(m1, res):
            self.assertTrue((x[0] > 0 and y > 0) or (x[0] < 0 and y < 0))

    def test_500x2b(self):
        """
        This test constructs a line of values in 2d space. 
        That if fit correctly maps perfectly to 1d space.
        The check is simply if the input value was positive
        then the output value should be similar.
        """
        m1 = self.generate_matrices_for_pca(30, seed=1304)
        X = Matrix(self.sds, m1)
        # print(features)
        X._check_matrix_op()
        params_dict = {'X': X, 'K': 1, 'scale': "FALSE", 'center': "FALSE"}
        nodeC = OperationNode2(self.sds, 'pca', named_input_nodes=params_dict,
                               outputs=[("res", OutputType.MATRIX), ("model", OutputType.MATRIX), ("scale", OutputType.MATRIX), ("center", OutputType.MATRIX)])
        nodeD = OperationNode2(self.sds, 'abs', unnamed_input_nodes=[nodeC.output_nodes["model"]])
        nodeD_res = nodeD.compute(verbose=True)

    def test_multiple_outputs(self):
        # Added a second test function because test_500x2b doesn't account for the case where multiple outputs of a node which provides
        # multiple outputs are used
        x = Matrix(self.sds, np.array([1,2,3,4,5,6,7,8,9]))
        y = Matrix(self.sds, np.array([10,20,30,40,50,60,70,80,90]))
        params_dict = {'X': x, 'Y': y}
        M2 = OperationNode2(self.sds, 'split', named_input_nodes=params_dict,
                            outputs=[("X_train", OutputType.MATRIX), ("X_test", OutputType.MATRIX), ("Y_train", OutputType.MATRIX), ("Y_test", OutputType.MATRIX)])
        M3 = M2.output_nodes["X_train"] + M2.output_nodes["Y_train"]
        res = M3.compute(verbose=True)

    def test_simple(self):
        """
        line of numbers. Here the pca should return values that are double or close to double of the last value
        """
        m1 = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        [res, model, _, _ ] = pca(Matrix(self.sds, m1), K=1,
                  scale=False, center=False).compute()
        for x in range(len(m1) - 1):
            self.assertTrue(abs(res[x + 1] - res[0] * (x + 2)) < 0.001)

    def generate_matrices_for_pca(self, dims: int, seed: int = 1234):
        np.random.seed(seed)

        mu, sigma = 0, 0.1
        s = np.random.normal(mu, sigma,  dims)

        m1 = np.array(np.c_[np.copy(s) * 1, np.copy(s)*0.3], dtype=np.double)

        return m1


if __name__ == "__main__":
    unittest.main(exit=False)
