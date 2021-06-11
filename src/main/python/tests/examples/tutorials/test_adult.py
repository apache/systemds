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
from systemds.operator import OperationNode
from systemds.operator.algorithm import kmeans, multiLogReg, multiLogRegPredict, l2svm, confusionMatrix
from systemds.script_building import DMLScript


class Test_DMLScript(unittest.TestCase):
    """
    Test class for adult dml script tutorial code.
    """

    sds: SystemDSContext = None
    d: DataManager = None
    base_path = "systemds/examples/tutorials/adult/"
    neural_net_src_path: str = "./tests/source/neural_net_source.dml"

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
        self.assertEqual((16281, 14), x_l.shape)

    def test_test_labels(self):
        y_l = self.d.get_test_labels()
        self.assertEqual((16281,), y_l.shape)

    def test_preprocess(self):
        #assumes certain preprocessing
        train_data, train_labels, test_data, test_labels = self.d.get_preprocessed_dataset()
        self.assertEqual((30162,104), train_data.shape)
        self.assertEqual((30162, ), train_labels.shape)
        self.assertEqual((15060,104), test_data.shape)
        self.assertEqual((15060, ), test_labels.shape)

    def test_multi_log_reg(self):
        # Reduced because we want the tests to finish a bit faster.
        train_count = 15000
        test_count = 5000

        train_data, train_labels, test_data, test_labels = self.d.get_preprocessed_dataset()

        # Train data
        X = self.sds.from_numpy( train_data[:train_count])
        Y = self.sds.from_numpy( train_labels[:train_count])
        Y = Y + 1.0

        # Test data
        Xt = self.sds.from_numpy(test_data[:test_count])
        Yt = self.sds.from_numpy(test_labels[:test_count])
        Yt = Yt + 1.0

        betas = multiLogReg(X, Y)

        [_, y_pred, acc] = multiLogRegPredict(Xt, betas, Yt).compute()

        self.assertGreater(acc, 80)

        confusion_matrix_abs, _ = confusionMatrix(self.sds.from_numpy(y_pred), Yt).compute()

        self.assertTrue(
            np.allclose(
                confusion_matrix_abs,
                np.array([[3503, 503],
                          [268, 726]])
            )
        )


    def test_multi_log_reg_interpolated_standardized(self):
        # Reduced because we want the tests to finish a bit faster.
        train_count = 15000
        test_count = 5000

        train_data, train_labels, test_data, test_labels = self.d.get_preprocessed_dataset(interpolate=True, standardize=True, dimred=0.1)

        # Train data
        X = self.sds.from_numpy( train_data[:train_count])
        Y = self.sds.from_numpy( train_labels[:train_count])
        Y = Y + 1.0

        # Test data
        Xt = self.sds.from_numpy(test_data[:test_count])
        Yt = self.sds.from_numpy(test_labels[:test_count])
        Yt = Yt + 1.0

        betas = multiLogReg(X, Y)

        [_, y_pred, acc] = multiLogRegPredict(Xt, betas, Yt).compute()

        self.assertGreater(acc, 80)
        
        confusion_matrix_abs, _ = confusionMatrix(self.sds.from_numpy(y_pred), Yt).compute()

        self.assertTrue(
            np.allclose(
                confusion_matrix_abs,
                np.array([[3583,  502],
                         [245,  670]])
            )
        )


    def test_neural_net(self):
        # Reduced because we want the tests to finish a bit faster.
        train_count = 15000
        test_count = 5000

        train_data, train_labels, test_data, test_labels = self.d.get_preprocessed_dataset(interpolate=True, standardize=True, dimred=0.1)

        # Train data
        X = self.sds.from_numpy( train_data[:train_count])
        Y = self.sds.from_numpy( train_labels[:train_count])

        # Test data
        Xt = self.sds.from_numpy(test_data[:test_count])
        Yt = self.sds.from_numpy(test_labels[:test_count])

        FFN_package = self.sds.source(self.neural_net_src_path, "fnn", print_imported_methods=True)

        network = FFN_package.train(X, Y, 1, 16, 0.01, 1)

        self.assertTrue(type(network) is not None) # sourcing and training seems to works

        FFN_package.save_model(network, '"model/python_FFN/"').compute(verbose=True)

        # TODO This does not work yet, not sure what the problem is
        # probs = FFN_package.predict(Xt, network).compute(True)
        # FFN_package.eval(Yt, Yt).compute()

if __name__ == "__main__":
    unittest.main(exit=False)
