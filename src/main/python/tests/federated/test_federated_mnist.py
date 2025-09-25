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
import itertools
import io
import json
import pandas as pd
from os import path, makedirs

from systemds.context import SystemDSContext
from systemds.examples.tutorials.mnist import DataManager
from systemds.operator.algorithm import kmeans, multiLogReg, multiLogRegPredict


def create_row_federated_dataset(name, dataset, num_parts=2, federated_workers=None):
    if federated_workers is None:
        federated_workers = ["localhost:8001", "localhost:8002"]
    tempdir = "./tests/federated/tmp/test_federated_mnist/"
    federated_file = path.join(tempdir, f"{name}.fed")
    makedirs(tempdir, exist_ok=True)

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
            "rows": rows,
            "cols": c,
            "data_type": "matrix",
            "value_type": "double",
        }

        dataset_part = dataset[rows_processed : rows_processed + rows]
        pd.DataFrame(dataset_part).to_csv(dataset_part_path, index=False, header=False)
        with io.open(f"{dataset_part_path}.mtd", "w", encoding="utf-8") as f:
            json.dump(mtd, f, ensure_ascii=False)

        fed_file_content.append(
            {
                "address": address,
                "dataType": "MATRIX",
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
                "data_type": "matrix",
                "value_type": "double",
            },
            f,
        )

    return federated_file


class TestFederatedMnist(unittest.TestCase):
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

    def test_multi_log_reg(self):
        # Reduced because we want the tests to finish a bit faster.
        train_count = 15000
        test_count = 5000
        # Train data
        np_train_data = self.d.get_train_data().reshape(60000, 28 * 28)[0:train_count]
        data_path_train = create_row_federated_dataset("train_data", np_train_data)
        X = self.sds.read(data_path_train)
        Y = self.sds.from_numpy(self.d.get_train_labels()[:train_count])
        Y = Y + 1.0

        # Test data
        np_test_data = self.d.get_test_data().reshape(10000, 28 * 28)[0:test_count]
        data_path_test = create_row_federated_dataset("test_data", np_test_data)
        Xt = self.sds.read(data_path_test)
        Yt = self.sds.from_numpy(self.d.get_test_labels()[:test_count])
        Yt = Yt + 1.0

        bias = multiLogReg(X, Y)

        with self.sds.capture_stats_context():
            [_, _, acc] = multiLogRegPredict(Xt, bias, Y=Yt).compute()
        stats = self.sds.take_stats()
        for fed_instr in [
            "fed_contains",
            "fed_*",
            "fed_-",
            "fed_uark+",
            "fed_r'",
            "fed_rightIndex",
        ]:
            self.assertIn(fed_instr, stats)
        self.assertGreater(acc, 80)


if __name__ == "__main__":
    unittest.main(exit=False)
