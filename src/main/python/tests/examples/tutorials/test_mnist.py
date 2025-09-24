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
from systemds.examples.tutorials.mnist import DataManager
from systemds.operator.algorithm import multiLogReg, multiLogRegPredict


class Test_DMLScript(unittest.TestCase):
    """
    Test class for mnist dml script tutorial code.
    """

    sds: SystemDSContext = None
    d: DataManager = None
    base_path = "systemds/examples/tutorials/mnist/"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)
        cls.d = DataManager()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_train_data(self):
        x = self.d.get_train_data()
        self.assertEqual((60000, 28, 28), x.shape)

    def test_train_labels(self):
        y = self.d.get_train_labels()
        self.assertEqual((60000,), y.shape)

    def test_test_data(self):
        x_l = self.d.get_test_data()
        self.assertEqual((10000, 28, 28), x_l.shape)

    def test_test_labels(self):
        y_l = self.d.get_test_labels()
        self.assertEqual((10000,), y_l.shape)

    def test_multi_log_reg(self):
        # Reduced because we want the tests to finish a bit faster.
        train_count = 5000
        test_count = 2000
        # Train data
        X = self.sds.from_numpy(
            self.d.get_train_data().reshape((60000, 28 * 28))[:train_count]
        )
        Y = self.sds.from_numpy(self.d.get_train_labels()[:train_count])
        Y = Y + 1.0

        # Test data
        Xt = self.sds.from_numpy(
            self.d.get_test_data().reshape((10000, 28 * 28))[:test_count]
        )
        Yt = self.sds.from_numpy(self.d.get_test_labels()[:test_count])
        Yt = Yt + 1.0

        bias = multiLogReg(X, Y, verbose=False)
        [_, _, acc] = multiLogRegPredict(Xt, bias, Y=Yt, verbose=False).compute()

        self.assertGreater(acc, 80)

    def test_multi_log_reg_with_read(self):
        train_count = 100
        test_count = 100
        X = self.sds.from_numpy(
            self.d.get_train_data().reshape((60000, 28 * 28))[:train_count]
        )
        X.write(self.base_path + "train_data").compute()
        Y = self.sds.from_numpy(self.d.get_train_labels()[:train_count]) + 1
        Y.write(self.base_path + "train_labels").compute()

        Xr = self.sds.read(self.base_path + "train_data")
        Yr = self.sds.read(self.base_path + "train_labels")

        bias = multiLogReg(Xr, Yr, verbose=False)
        # Test data
        Xt = self.sds.from_numpy(
            self.d.get_test_data().reshape((10000, 28 * 28))[:test_count]
        )
        Yt = self.sds.from_numpy(self.d.get_test_labels()[:test_count])
        Yt = Yt + 1.0

        [_, _, acc] = multiLogRegPredict(Xt, bias, Y=Yt).compute(verbose=False)

        self.assertGreater(acc, 70)


if __name__ == "__main__":
    unittest.main(exit=False)
