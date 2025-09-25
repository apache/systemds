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
import itertools
import shutil
import unittest
import io
import json
import pandas as pd
from os import path, makedirs

from systemds.context import SystemDSContext
from systemds.examples.tutorials.adult import DataManager
from systemds.operator.algorithm.builtin.scale import scale
from systemds.operator.algorithm.builtin.scaleApply import scaleApply


def create_schema(dataset):
    schema = []
    for dtype in dataset.dtypes:
        if pd.api.types.is_integer_dtype(dtype):
            schema.append("int64")
        elif pd.api.types.is_float_dtype(dtype):
            schema.append("fp64")
        elif pd.api.types.is_bool_dtype(dtype):
            schema.append("bool")
        else:
            schema.append("string")
    return ",".join(schema)


def create_row_federated_dataset(name, dataset, num_parts=2, federated_workers=None):
    if federated_workers is None:
        federated_workers = ["localhost:8001", "localhost:8002"]
    tempdir = "./tests/federated/tmp/test_federated_adult_neural/"
    federated_file = path.join(tempdir, f"{name}.fed")
    makedirs(tempdir, exist_ok=True)

    schema = create_schema(dataset)
    r = dataset.shape[0] // num_parts
    rs = [r for _ in range(num_parts - 1)] + [dataset.shape[0] - r * (num_parts - 1)]
    c = dataset.shape[1]

    fed_file_content = []
    rows_processed = 0
    for worker_id, address, rows in zip(
        range(num_parts), itertools.cycle(federated_workers), rs
    ):
        dataset_part_path = path.join(tempdir, f"{name}{worker_id}.csv")
        mtd = {
            "format": "csv",
            "header": True,
            "rows": rows,
            "cols": c,
            "data_type": "frame",
            "schema": schema,
        }

        dataset_part = dataset[rows_processed : rows_processed + rows]
        dataset_part.to_csv(dataset_part_path, index=False)
        with io.open(f"{dataset_part_path}.mtd", "w", encoding="utf-8") as f:
            json.dump(mtd, f, ensure_ascii=False)

        fed_file_content.append(
            {
                "address": address,
                "dataType": "FRAME",
                "filepath": dataset_part_path,
                "begin": [rows_processed, 0],
                "end": [rows_processed + rows, c],
            }
        )
        rows_processed += rows

    with open(federated_file, "w", encoding="utf-8") as f:
        json.dump(fed_file_content, f)
    with open(federated_file + ".mtd", "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "federated",
                "rows": dataset.shape[0],
                "cols": c,
                "data_type": "frame",
                "schema": schema,
            },
            f,
        )

    return federated_file


class TestFederatedAdultNeural(unittest.TestCase):
    """
    Test class for adult neural network code
    """

    sds: SystemDSContext = None
    d: DataManager = None
    neural_net_src_path: str = "tests/examples/tutorials/neural_net_source.dml"
    preprocess_src_path: str = "tests/examples/tutorials/preprocess.dml"
    data_path_train: str = ""
    data_path_test: str = ""
    labels_path_train: str = ""
    labels_path_test: str = ""
    dataset_jspec: str = "../../test/resources/datasets/adult/jspec.json"

    train_count: int = 15000
    test_count: int = 300

    network_dir: str = "tests/examples/tutorials/model"
    network: str = network_dir + "/fnn"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)
        cls.d = DataManager()
        cls.data_path_train = create_row_federated_dataset(
            "train_data", cls.d.get_train_data_pandas()[0 : cls.train_count]
        )
        cls.labels_path_train = create_row_federated_dataset(
            "train_labels", cls.d.get_train_labels_pandas()[0 : cls.train_count]
        )
        cls.data_path_test = create_row_federated_dataset(
            "test_data", cls.d.get_test_data_pandas()[0 : cls.test_count]
        )
        cls.labels_path_test = create_row_federated_dataset(
            "test_labels", cls.d.get_test_labels_pandas()[0 : cls.test_count]
        )
        shutil.rmtree(cls.network_dir, ignore_errors=True)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()
        shutil.rmtree(cls.network_dir, ignore_errors=True)

    # Tests

    @unittest.skip("`toOneHot()` won't be federated -> param-server won't work")
    def test_train_neural_net(self):
        self.train_neural_net_and_save()
        self.eval_neural_net()

    @unittest.skip("`toOneHot()` won't be federated -> param-server won't work")
    def test_train_predict(self):
        self.train_neural_net_and_predict()

    # Helper methods

    def prepare_x(self):
        jspec = self.d.get_jspec(self.sds)
        train_x_frame = self.sds.read(self.data_path_train)
        train_x, M1 = train_x_frame.transform_encode(spec=jspec)
        test_x_frame = self.sds.read(self.data_path_test)
        test_x = test_x_frame.transform_apply(spec=jspec, meta=M1)
        # Scale and shift .... not needed because of sigmoid layer,
        # could be useful therefore tested.
        [train_x, ce, sc] = scale(train_x)
        test_x = scaleApply(test_x, ce, sc)
        return [train_x, test_x]

    def prepare_y(self):
        jspec_dict = {"recode": ["income"]}
        jspec_labels = self.sds.scalar(f'"{jspec_dict}"')
        train_y_frame = self.sds.read(self.labels_path_train)
        train_y, M2 = train_y_frame.transform_encode(spec=jspec_labels)
        test_y_frame = self.sds.read(self.labels_path_test)
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
        network = FFN_package.train_paramserv(train_x, train_y, 1, 16, 0.01, 2, 1)
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
