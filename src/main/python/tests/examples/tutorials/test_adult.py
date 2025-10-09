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
from systemds.operator.algorithm import confusionMatrix, multiLogReg, multiLogRegPredict


class TestAdultStandardML(unittest.TestCase):
    """
    Test class for adult dml script tutorial code.
    """

    sds: SystemDSContext = None
    d: DataManager = None
    neural_net_src_path: str = "tests/examples/tutorials/neural_net_source.dml"
    preprocess_src_path: str = "tests/examples/tutorials/preprocess.dml"
    dataset_path_train: str = "../../test/resources/datasets/adult/train_data.csv"
    dataset_path_train_mtd: str = (
        "../../test/resources/datasets/adult/train_data.csv.mtd"
    )
    dataset_path_test: str = "../../test/resources/datasets/adult/test_data.csv"
    dataset_path_test_mtd: str = "../../test/resources/datasets/adult/test_data.csv.mtd"
    dataset_jspec: str = "../../test/resources/datasets/adult/jspec.json"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)
        cls.d = DataManager()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_train_data(self):
        x = self.d.get_train_data_pandas()
        self.assertEqual((32561, 14), x.shape)

    def test_train_labels(self):
        y = self.d.get_train_labels_pandas()
        self.assertEqual((32561, 1), y.shape)

    def test_test_data(self):
        x_l = self.d.get_test_data_pandas()
        self.assertEqual((16281, 14), x_l.shape)

    def test_test_labels(self):
        y_l = self.d.get_test_labels_pandas()
        self.assertEqual((16281, 1), y_l.shape)

    def test_train_data_pandas_vs_systemds(self):
        pandas = self.d.get_train_data_pandas()[0:2000]
        systemds = self.d.get_train_data(self.sds)[0:2000].compute()
        self.assertTrue(len(pandas.columns.difference(systemds.columns)) == 0)
        self.assertEqual(pandas.shape, systemds.shape)

    def test_train_labels_pandas_vs_systemds(self):
        # Pandas does not strip the parsed values.. so i have to do it here.
        pandas = np.array(
            [
                x.strip()
                for x in self.d.get_train_labels_pandas()[0:2000].to_numpy().flatten()
            ]
        )
        systemds = (
            self.d.get_train_labels(self.sds)[0:2000].compute().to_numpy().flatten()
        )
        comp = pandas == systemds
        self.assertTrue(comp.all())

    def test_test_labels_pandas_vs_systemds(self):
        # Pandas does not strip the parsed values.. so i have to do it here.
        pandas = np.array(
            [
                x.strip()
                for x in self.d.get_test_labels_pandas()[0:2000].to_numpy().flatten()
            ]
        )
        systemds = (
            self.d.get_test_labels(self.sds)[0:2000].compute().to_numpy().flatten()
        )
        comp = pandas == systemds
        self.assertTrue(comp.all())

    def test_transform_encode_train_data(self):
        jspec = self.d.get_jspec(self.sds)
        train_x, M1 = self.d.get_train_data(self.sds)[0:2000].transform_encode(
            spec=jspec
        )
        train_x_numpy = train_x.compute()
        self.assertEqual((2000, 101), train_x_numpy.shape)

    def test_transform_encode_apply_test_data(self):
        jspec = self.d.get_jspec(self.sds)
        train_x, M1 = self.d.get_train_data(self.sds)[0:2000].transform_encode(
            spec=jspec
        )
        test_x = self.d.get_test_data(self.sds)[0:2000].transform_apply(
            spec=jspec, meta=M1
        )
        test_x_numpy = test_x.compute()
        self.assertEqual((2000, 101), test_x_numpy.shape)

    def test_transform_encode_train_labels(self):
        jspec_dict = {"recode": ["income"]}
        jspec = self.sds.scalar(f'"{jspec_dict}"')
        train_y, M1 = self.d.get_train_labels(self.sds)[0:2000].transform_encode(
            spec=jspec
        )
        train_y_numpy = train_y.compute()
        self.assertEqual((2000, 1), train_y_numpy.shape)

    def test_transform_encode_test_labels(self):
        jspec_dict = {"recode": ["income"]}
        jspec = self.sds.scalar(f'"{jspec_dict}"')
        train_y, M1 = self.d.get_train_labels(self.sds)[0:2000].transform_encode(
            spec=jspec
        )
        test_y = self.d.get_test_labels(self.sds)[0:2000].transform_apply(
            spec=jspec, meta=M1
        )
        test_y_numpy = test_y.compute()
        self.assertEqual((2000, 1), test_y_numpy.shape)

    def test_multi_log_reg(self):
        # Reduced because we want the tests to finish a bit faster.
        train_count = 2000
        test_count = 500

        jspec_data = self.d.get_jspec(self.sds)
        train_x_frame = self.d.get_train_data(self.sds)[0:train_count]
        train_x, M1 = train_x_frame.transform_encode(spec=jspec_data)
        test_x_frame = self.d.get_test_data(self.sds)[0:test_count]
        test_x = test_x_frame.transform_apply(spec=jspec_data, meta=M1)

        jspec_dict = {"recode": ["income"]}
        jspec_labels = self.sds.scalar(f'"{jspec_dict}"')
        train_y_frame = self.d.get_train_labels(self.sds)[0:train_count]
        train_y, M2 = train_y_frame.transform_encode(spec=jspec_labels)
        test_y_frame = self.d.get_test_labels(self.sds)[0:test_count]
        test_y = test_y_frame.transform_apply(spec=jspec_labels, meta=M2)

        betas = multiLogReg(train_x, train_y, verbose=False)
        [_, y_pred, acc] = multiLogRegPredict(test_x, betas, Y=test_y, verbose=False)

        [_, conf_avg] = confusionMatrix(y_pred, test_y)
        confusion_numpy = conf_avg.compute()

        self.assertTrue(confusion_numpy[0][0] > 0.8)
        self.assertTrue(confusion_numpy[0][1] < 0.5)
        self.assertTrue(confusion_numpy[1][1] > 0.5)
        self.assertTrue(confusion_numpy[1][0] < 0.2)


if __name__ == "__main__":
    unittest.main(exit=False)
