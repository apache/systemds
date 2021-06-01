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
from systemds.operator.algorithm import pca

from systemds.operator import List
from systemds.script_building.dag import OutputType


class TestListOperations(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_creation(self):
        """
        Tests the creation of a List object via the SystemDSContext
        """
        m1 = self.sds.from_numpy(np.array([1, 2, 3]))
        m2 = self.sds.from_numpy(np.array([4, 5, 6]))
        list_obj = self.sds.list(m1, m2)
        tmp = list_obj[0] + list_obj[1]
        res = tmp.compute()
        self.assertTrue(np.allclose(m2, res))

    def test_addition(self):
        """
        Tests the creation of a List object via the SystemDSContext and adds a value
        """
        m1 = self.sds.from_numpy(np.array([1, 2, 3]))
        m2 = self.sds.from_numpy(np.array([4, 5, 6]))
        list_obj = self.sds.list(m1, m2)
        tmp = list_obj[0] + 2
        res = tmp.compute()
        self.assertTrue(np.allclose(m2 + 2, res))

    def test_500x2b(self):
        """
        The purpose of this test is to show that an operation can be performed on the output of a multi output list node,
        without the need of calculating the result first.
        """
        m1 = self.generate_matrices_for_pca(30, seed=1304)
        node0 = self.sds.from_numpy(m1)
        # print(features)
        node1 = List(node0.sds_context, 'pca', named_input_nodes={"X": node0, "K": 1, "scale": "FALSE", "center": "FALSE"},
                     outputs=[("res", OutputType.MATRIX), ("model", OutputType.MATRIX), ("scale", OutputType.MATRIX), ("center", OutputType.MATRIX)])
        node2 = node1["res"].abs()
        res = node2.compute(verbose=False)

    def test_multiple_outputs(self):
        """
        The purpose of this test is to show that we can use multiple outputs
        of a single list node in the DAG in one script
        """
        node0 = self.sds.from_numpy(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        node1 = self.sds.from_numpy(np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]))
        params_dict = {'X': node0, 'Y': node1}
        node2 = List(self.sds, 'split', named_input_nodes=params_dict,
                     outputs=[("X_train", OutputType.MATRIX), ("X_test", OutputType.MATRIX), ("Y_train", OutputType.MATRIX), ("Y_test", OutputType.MATRIX)])
        node3 = node2["X_train"] + node2["Y_train"]
        # X_train and Y_train are of the same shape because node0 and node1 have both only one dimension.
        # Therefore they can be added together
        res = node3.compute(verbose=False)

    def generate_matrices_for_pca(self, dims: int, seed: int = 1234):
        np.random.seed(seed)

        mu, sigma = 0, 0.1
        s = np.random.normal(mu, sigma,  dims)

        m1 = np.array(np.c_[np.copy(s) * 1, np.copy(s)*0.3], dtype=np.double)

        return m1


if __name__ == "__main__":
    unittest.main(exit=False)
