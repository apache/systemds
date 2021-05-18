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
from systemds.examples.tutorials.adult import DataManager
from systemds.operator.algorithm import kmeans, multiLogReg, multiLogRegPredict
from systemds.script_building import DMLScript


class Test_DMLScript(unittest.TestCase):
    """
    Test class for mnist dml script tutorial code.
    """

    sds: SystemDSContext = None
    d: DataManager = None
    base_path = "systemds/examples/tutorials/adult/"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()
        cls.d = DataManager()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_train_data(self):
        x = self.d.get_train_data()
        self.assertEqual((32561, 14), x.shape)

    def test_train_labels(self):
        y = self.d.get_train_labels()
        self.assertEqual((32561,), y.shape)

    def test_test_data(self):
        x_l = self.d.get_test_data()
        self.assertEqual((16282, 14), x_l.shape)

    def test_test_labels(self):
        y_l = self.d.get_test_labels()
        self.assertEqual((16282,), y_l.shape)

    def test_preprocess(self):
        #assumes certain preprocessing
        train_data, train_labels, test_data, test_labels = self.d.get_preprocessed_dataset()
        self.assertEqual((30162,104), train_data.shape)
        self.assertEqual((30162, ), train_labels.shape)
        self.assertEqual((15061,104), test_data.shape)
        self.assertEqual((15061, ), test_labels.shape)


if __name__ == "__main__":
    unittest.main(exit=False)
