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

m1 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int16)
m2 = np.asarray([[2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.int16)

tempdir = "./tests/federated/tmp/test_federated_aggregations_noHeader/"
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
np.savetxt(tempdir + "m1.csv", m1, delimiter=",", fmt="%d")
with io.open(tempdir + "m1.csv.mtd", "w", encoding="utf-8") as f:
    f.write(json.dumps(mtd, ensure_ascii=False))

np.savetxt(tempdir + "m2.csv", m2, delimiter=",", fmt="%d")
with io.open(tempdir + "m2.csv.mtd", "w", encoding="utf-8") as f:
    f.write(json.dumps(mtd, ensure_ascii=False))

# Federated workers + file locations
fed1 = "localhost:8001/" + tempdir + "m1.csv"
fed2 = "localhost:8002/" + tempdir + "m2.csv"


class TestFederatedAggFn(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_equals(self):
        f_m = self.sds.federated([fed1], [([0, 0], [dim, dim])]).compute()
        self.assertTrue(np.allclose(f_m, m1))

    def test_sum3(self):
        #   [[m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]]
        f_m_a = (
            self.sds.federated(
                [fed1, fed2], [([0, 0], [dim, dim]), ([0, dim], [dim, dim * 2])]
            )
            .sum()
            .compute()
        )
        m1_m2 = m1.sum() + m2.sum()
        self.assertAlmostEqual(f_m_a, m1_m2)

    def test_sum1(self):
        f_m1 = self.sds.federated([fed1], [([0, 0], [dim, dim])]).sum().compute()
        m1_r = m1.sum()
        self.assertAlmostEqual(f_m1, m1_r)

    def test_sum2(self):
        f_m2 = self.sds.federated([fed2], [([0, 0], [dim, dim])]).sum().compute()
        m2_r = m2.sum()
        self.assertAlmostEqual(f_m2, m2_r)

    def test_sum3(self):
        #   [[m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]]
        f_m1_m2 = (
            self.sds.federated(
                [fed1, fed2], [([0, 0], [dim, dim]), ([0, dim], [dim, dim * 2])]
            )
            .sum()
            .compute()
        )

        m1_m2 = np.concatenate((m1, m2), axis=1).sum()

        self.assertAlmostEqual(f_m1_m2, m1_m2)

    def test_sum4(self):
        #   [[m1,m1,m1,m1,m1]
        #    [m1,m1,m1,m1,m1]
        #    [m1,m1,m1,m1,m1]
        #    [m1,m1,m1,m1,m1]
        #    [m1,m1,m1,m1,m1]
        #    [m2,m2,m2,m2,m2]
        #    [m2,m2,m2,m2,m2]
        #    [m2,m2,m2,m2,m2]
        #    [m2,m2,m2,m2,m2]
        #    [m2,m2,m2,m2,m2]]
        f_m1_m2 = (
            self.sds.federated(
                [fed1, fed2], [([0, 0], [dim, dim]), ([dim, 0], [dim * 2, dim])]
            )
            .sum()
            .compute()
        )
        m1_m2 = np.concatenate((m1, m2)).sum()
        self.assertAlmostEqual(f_m1_m2, m1_m2)

    # -----------------------------------
    # The rest of the tests are
    # Extended functionality not working Yet
    # -----------------------------------

    def test_sum5(self):
        #   [[m1,m1,m1,m1,m1, 0, 0, 0, 0, 0]
        #    [m1,m1,m1,m1,m1, 0, 0, 0, 0, 0]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [ 0, 0, 0, 0, 0,m2,m2,m2,m2,m2]
        #    [ 0, 0, 0, 0, 0,m2,m2,m2,m2,m2]]
        f_m_a = (
            self.sds.federated(
                [fed1, fed2], [([0, 0], [dim, dim]), ([2, dim], [dim + 2, dim * 2])]
            )
            .sum()
            .compute()
        )
        m1_m2 = m1.sum() + m2.sum()
        self.assertAlmostEqual(f_m_a, m1_m2)

    def test_sum8(self):
        #   [[ 0, 0, 0, 0, 0, 0, 0, 0]
        #    [ 0, 0, 0, 0, 0, 0, 0, 0]
        #    [ 0, 0, 0,m1,m1,m1,m1,m1]
        #    [ 0, 0, 0,m1,m1,m1,m1,m1]
        #    [ 0, 0, 0,m1,m1,m1,m1,m1]
        #    [ 0, 0, 0,m1,m1,m1,m1,m1]
        #    [ 0, 0, 0,m1,m1,m1,m1,m1]]
        f_m_a = (
            self.sds.federated([fed1], [([2, 3], [dim + 2, dim + 3])]).sum().compute()
        )

        m = m1.sum()

        self.assertAlmostEqual(f_m_a, m)


if __name__ == "__main__":
    unittest.main(exit=False)
