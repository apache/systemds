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

import shutil
import unittest

from systemds.context import SystemDSContext
from systemds.examples.tutorials.adult import DataManager
from systemds.operator.algorithm.builtin.scale import scale
from systemds.operator.algorithm.builtin.scaleApply import scaleApply


class TestAdultNeural(unittest.TestCase):
    """
    Test class for adult neural network code
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

    train_count: int = 5000
    test_count: int = 300

    network_dir: str = "tests/examples/tutorials/model"
    network: str = network_dir + "/fnn"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)
        cls.d = DataManager()
        shutil.rmtree(cls.network_dir, ignore_errors=True)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()
        shutil.rmtree(cls.network_dir, ignore_errors=True)

    # Tests

    def test_train_neural_net(self):
        self.train_neural_net_and_save()
        self.eval_neural_net()

    def test_train_predict(self):
        self.train_neural_net_and_predict()

    # Helper methods

    def prepare_x(self):
        jspec = self.d.get_jspec(self.sds)
        train_x_frame = self.d.get_train_data(self.sds)[0 : self.train_count]
        train_x, M1 = train_x_frame.transform_encode(spec=jspec)
        test_x_frame = self.d.get_test_data(self.sds)[0 : self.test_count]
        test_x = test_x_frame.transform_apply(spec=jspec, meta=M1)
        # Scale and shift .... not needed because of sigmoid layer,
        # could be useful therefore tested.
        [train_x, ce, sc] = scale(train_x)
        test_x = scaleApply(test_x, ce, sc)
        return [train_x, test_x]

    def prepare_y(self):
        jspec_dict = {"recode": ["income"]}
        jspec_labels = self.sds.scalar(f'"{jspec_dict}"')
        train_y_frame = self.d.get_train_labels(self.sds)[0 : self.train_count]
        train_y, M2 = train_y_frame.transform_encode(spec=jspec_labels)
        test_y_frame = self.d.get_test_labels(self.sds)[0 : self.test_count]
        test_y = test_y_frame.transform_apply(spec=jspec_labels, meta=M2)
        labels = 2
        train_y = train_y.to_one_hot(labels)
        test_y = test_y.to_one_hot(labels)
        return [train_y, test_y]

    def prepare(self):
        x = self.prepare_x()
        y = self.prepare_y()
        return [x[0], x[1], y[0], y[1]]

    def train_neural_net_and_save(self):
        [train_x, _, train_y, _] = self.prepare()
        FFN_package = self.sds.source(self.neural_net_src_path, "fnn")
        network = FFN_package.train(train_x, train_y, 4, 16, 0.01, 1)
        network.write(self.network).compute()

    def train_neural_net_and_predict(self):
        [train_x, test_x, train_y, test_y] = self.prepare()
        FFN_package = self.sds.source(self.neural_net_src_path, "fnn")
        network = FFN_package.train_paramserv(train_x, train_y, 4, 16, 0.01, 2, 1)
        probs = FFN_package.predict(test_x, network)
        accuracy = FFN_package.eval(probs, test_y).compute()
        # accuracy is returned in percent
        self.assertTrue(accuracy > 0.80)

    def eval_neural_net(self):
        [_, test_x, _, test_y] = self.prepare()
        network = self.sds.read(self.network)
        FFN_package = self.sds.source(self.neural_net_src_path, "fnn")
        probs = FFN_package.predict(test_x, network)
        accuracy = FFN_package.eval(probs, test_y).compute()
        # accuracy is returned in percent
        self.assertTrue(accuracy > 0.80)


if __name__ == "__main__":
    unittest.main(exit=False)
