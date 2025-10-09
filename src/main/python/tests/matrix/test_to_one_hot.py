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


class TestMatrixOneHot(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_one_hot_1(self):
        m1 = np.array([1])
        res = self.sds.from_numpy(m1).to_one_hot(3).compute()
        self.assertTrue((res == [[1, 0, 0]]).all())

    def test_one_hot_2(self):
        m1 = np.array([2])
        res = self.sds.from_numpy(m1).to_one_hot(3).compute()
        self.assertTrue((res == [[0, 1, 0]]).all())

    def test_one_hot_3(self):
        m1 = np.array([2])
        res = self.sds.from_numpy(m1).to_one_hot(2).compute()
        self.assertTrue((res == [[0, 1]]).all())

    def test_one_hot_2_2(self):
        m1 = np.array([2, 2])
        res = self.sds.from_numpy(m1).to_one_hot(2).compute()
        self.assertTrue((res == [[0, 1], [0, 1]]).all())

    def test_one_hot_1_2(self):
        m1 = np.array([1, 2])
        res = self.sds.from_numpy(m1).to_one_hot(2).compute()
        self.assertTrue((res == [[1, 0], [0, 1]]).all())

    def test_one_hot_1_2(self):
        m1 = np.array([1, 2, 2])
        res = self.sds.from_numpy(m1).to_one_hot(2).compute()
        self.assertTrue((res == [[1, 0], [0, 1], [0, 1]]).all())

    # TODO make tests for runtime errors, like this one
    # def test_neg_one_hot_toHighValue(self):
    #     m1 = np.array([3])
    #     with self.assertRaises(ValueError) as context:
    #         res = self.sds.from_numpy( m1).to_one_hot(2).compute()

    def test_one_hot_matrix_1(self):
        m1 = np.array([[1], [2], [3]])
        res = self.sds.from_numpy(m1).to_one_hot(3).compute()
        self.assertTrue((res == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]).all())

    def test_one_hot_matrix_2(self):
        m1 = np.array([[1], [3], [3]])
        res = self.sds.from_numpy(m1).to_one_hot(3).compute()
        self.assertTrue((res == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]).all())

    def test_one_hot_matrix_3(self):
        m1 = np.array([[1], [2], [1]])
        res = self.sds.from_numpy(m1).to_one_hot(2).compute()
        self.assertTrue((res == [[1, 0], [0, 1], [1, 0]]).all())

    def test_neg_one_hot_numClasses(self):
        m1 = np.array([1])
        with self.assertRaises(ValueError) as context:
            res = self.sds.from_numpy(m1).to_one_hot(1).compute()

    def test_neg_one_hot_inputShape(self):
        m1 = np.array([[1]])
        with self.assertRaises(ValueError) as context:
            res = self.sds.from_numpy(m1).to_one_hot(1).compute()


if __name__ == "__main__":
    unittest.main(exit=False)
