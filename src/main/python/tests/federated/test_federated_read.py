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

import io
import json
import os
import shutil
import sys
import unittest

import numpy as np
from systemds.context import SystemDSContext

os.environ["SYSDS_QUIET"] = "1"

dim = 3

m = np.reshape(np.arange(1, dim * dim + 1, 1), (dim, dim))

tempdir = "./tests/federated/tmp/test_federated_matrixmult/"
mtd = {
    "format": "csv",
    "header": False,
    "rows": dim,
    "cols": dim,
    "data_type": "matrix",
    "value_type": "double",
}

# Create the testing directory if it does not exist.
if not os.path.exists(tempdir):
    os.makedirs(tempdir)

# Save data files for the Federated workers.
np.savetxt(tempdir + "m.csv", m, delimiter=",")
with io.open(tempdir + "m.csv.mtd", "w", encoding="utf-8") as f:
    f.write(json.dumps(mtd, ensure_ascii=False))

# Federated workers + file locations
fed1 = "localhost:8001/" + tempdir + "m.csv"
fed2 = "localhost:8002/" + tempdir + "m.csv"
fed3 = "localhost:8003/" + tempdir + "m.csv"

fed1_file = tempdir + "m1.fed"
fed2_file = tempdir + "m2.fed"
fed3_file = tempdir + "m3.fed"


class TestFederatedAggFn(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()
        cls.sds.federated([fed1], [([0, 0], [dim, dim])]).write(
            fed1_file, format="federated"
        ).compute()
        cls.sds.federated(
            [fed1, fed2], [([0, 0], [dim, dim]), ([0, dim], [dim, dim * 2])]
        ).write(fed2_file, format="federated").compute()
        cls.sds.federated(
            [fed1, fed2, fed3],
            [
                ([0, 0], [dim, dim]),
                ([0, dim], [dim, dim * 2]),
                ([0, dim * 2], [dim, dim * 3]),
            ],
        ).write(fed3_file, format="federated").compute()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_verify_same_input(self):
        f_m = self.sds.federated([fed1], [([0, 0], [dim, dim])]).compute()
        self.assertTrue(np.allclose(f_m, m))

    def test_verify_same_input_if_reading_fed(self):
        f_m = self.sds.read(fed1_file).compute()
        self.assertTrue(np.allclose(f_m, m))

    def test_verify_same_input_if_reading_fed2(self):
        f_m = self.sds.read(fed2_file).compute()
        m2 = np.column_stack((m, m))
        self.assertTrue(np.allclose(f_m, m2))

    def test_verify_same_input_if_reading_fed3(self):
        f_m = self.sds.read(fed3_file).compute()
        m2 = np.column_stack((m, m))
        m3 = np.column_stack((m, m2))
        self.assertTrue(np.allclose(f_m, m3))


if __name__ == "__main__":
    unittest.main(exit=False)
