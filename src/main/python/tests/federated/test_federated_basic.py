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

dim = 5
np.random.seed(132)
m1 = np.array(np.random.randint(100, size=dim * dim) + 1.01, dtype=np.double)
m1.shape = (dim, dim)
m2 = np.array(np.random.randint(5, size=dim * dim) + 1, dtype=np.double)
m2.shape = (dim, dim)

tempdir = "./tests/federated/tmp/test_federated_aggregations/"
mtd = {
    "format": "csv",
    "header": True,
    "rows": dim,
    "cols": dim,
    "data_type": "matrix",
    "value_type": "double",
}

# Create the testing directory if it does not exist.
if not os.path.exists(tempdir):
    os.makedirs(tempdir)

# Save data files for the Federated workers.
np.savetxt(tempdir + "m1.csv", m1, delimiter=",", header="a,b,c,d,e")
with io.open(tempdir + "m1.csv.mtd", "w", encoding="utf-8") as f:
    f.write(json.dumps(mtd, ensure_ascii=False))

np.savetxt(tempdir + "m2.csv", m2, delimiter=",", header="a,b,c,d,e")
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

    def test_1(self):
        f_m1 = self.sds.federated([fed1], [([0, 0], [dim, dim])]).compute()
        res = np.allclose(f_m1, m1)
        self.assertTrue(res, "\n" + str(f_m1) + " is not equal to \n" + str(m1))

    def test_2(self):
        f_m2 = self.sds.federated([fed2], [([0, 0], [dim, dim])]).compute()
        res = np.allclose(f_m2, m2)
        self.assertTrue(res)

    def test_3(self):
        #   [[m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]]
        f_m1_m2 = self.sds.federated(
            [fed1, fed2], [([0, 0], [dim, dim]), ([0, dim], [dim, dim * 2])]
        ).compute()
        m1_m2 = np.concatenate((m1, m2), axis=1)
        res = np.allclose(f_m1_m2, m1_m2)
        self.assertTrue(res)

    def test_4(self):
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
        f_m1_m2 = self.sds.federated(
            [fed1, fed2], [([0, 0], [dim, dim]), ([dim, 0], [dim * 2, dim])]
        ).compute()
        m1_m2 = np.concatenate((m1, m2))
        res = np.allclose(f_m1_m2, m1_m2)
        self.assertTrue(res)

    def test_5(self):
        #   [[m1,m1,m1,m1,m1, 0, 0, 0, 0, 0]
        #    [m1,m1,m1,m1,m1, 0, 0, 0, 0, 0]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
        #    [ 0, 0, 0, 0, 0,m2,m2,m2,m2,m2]
        #    [ 0, 0, 0, 0, 0,m2,m2,m2,m2,m2]]
        f_m1_m2 = self.sds.federated(
            [fed1, fed2], [([0, 0], [dim, dim]), ([2, dim], [dim + 2, dim * 2])]
        ).compute()

        m1_p = np.concatenate((m1, np.zeros((2, dim))))
        m2_p = np.concatenate((np.zeros((2, dim)), m2))
        m1_m2 = np.concatenate((m1_p, m2_p), axis=1)
        res = np.allclose(f_m1_m2, m1_m2)
        self.assertTrue(res)

    # def test_6(self):
    #     # Note it overwrites the value in the field. not sum or anything else.
    #     #   [[m1,m1,m1,m1,m1, 0, 0, 0]
    #     #    [m1,m1,m1,m1,m1, 0, 0, 0]
    #     #    [m1,m1,m1,m2,m2,m2,m2,m2]
    #     #    [m1,m1,m1,m2,m2,m2,m2,m2]
    #     #    [m1,m1,m1,m2,m2,m2,m2,m2]
    #     #    [ 0, 0, 0,m2,m2,m2,m2,m2]
    #     #    [ 0, 0, 0,m2,m2,m2,m2,m2]]
    #     f_m1_m2 = self.sds.federated(
    #         [fed1, fed2], [([0, 0], [dim, dim]), ([2, 3], [dim + 2, dim + 3])]
    #     ).compute()

    #     m1_m2 = np.zeros((dim + 2, dim + 3))
    #     m1_m2[0:dim, 0:dim] = m1
    #     m1_m2[2 : dim + 2, 3 : dim + 3] = m2

    #     res = np.allclose(f_m1_m2, m1_m2)
    #     self.assertTrue(res)

    # def test_7(self):
    #     #   [[m1,m1,m1,m1,m1, 0, 0, 0]
    #     #    [m1,m1,m1,m1,m1, 0, 0, 0]
    #     #    [m1,m1,m1,m2,m2,m2,m2,m2]
    #     #    [m1,m1,m1,m2,m2,m2,m2,m2]    +     1
    #     #    [m1,m1,m1,m2,m2,m2,m2,m2]
    #     #    [ 0, 0, 0,m2,m2,m2,m2,m2]
    #     #    [ 0, 0, 0,m2,m2,m2,m2,m2]]
    #     f_m1_m2 = self.sds.federated(
    #         [fed1, fed2], [([0, 0], [dim, dim]), ([2, 3], [dim + 2, dim + 3])]
    #     )
    #     f_m1_m2 = (f_m1_m2 + 1).compute()
    #     m1_m2 = np.zeros((dim + 2, dim + 3))
    #     m1_m2[0:dim, 0:dim] = m1
    #     m1_m2[2 : dim + 2, 3 : dim + 3] = m2
    #     m1_m2 += 1
    #     res = np.allclose(f_m1_m2, m1_m2)
    #     if not res:
    #         print("Federated:")
    #         print(f_m1_m2)
    #         print("numpy:")
    #         print(m1_m2)
    #     self.assertTrue(res)

    def test_8(self):
        #   [[ 0, 0, 0, 0, 0, 0, 0, 0]
        #    [ 0, 0, 0, 0, 0, 0, 0, 0]
        #    [ 0, 0, 0,m1,m1,m1,m1,m1]
        #    [ 0, 0, 0,m1,m1,m1,m1,m1]
        #    [ 0, 0, 0,m1,m1,m1,m1,m1]
        #    [ 0, 0, 0,m1,m1,m1,m1,m1]
        #    [ 0, 0, 0,m1,m1,m1,m1,m1]]
        f_m1_m2 = self.sds.federated([fed1], [([2, 3], [dim + 2, dim + 3])])
        f_m1_m2 = (f_m1_m2).compute()
        m1_m2 = np.zeros((dim + 2, dim + 3))
        m1_m2[2 : dim + 2, 3 : dim + 3] = m1
        res = np.allclose(f_m1_m2, m1_m2)
        if not res:
            print("Federated:")
            print(f_m1_m2)
            print("numpy:")
            print(m1_m2)
        self.assertTrue(res)

    # def test_9(self):
    #     #   [[ 0, 0, 0, 0, 0, 0, 0, 0]
    #     #    [ 0, 0, 0, 0, 0, 0, 0, 0]
    #     #    [ 0, 0, 0,m1,m1,m1,m1,m1]
    #     #    [ 0, 0, 0,m1,m1,m1,m1,m1]    +     1
    #     #    [ 0, 0, 0,m1,m1,m1,m1,m1]
    #     #    [ 0, 0, 0,m1,m1,m1,m1,m1]
    #     #    [ 0, 0, 0,m1,m1,m1,m1,m1]]
    #     f_m1_m2 = self.sds.federated( [fed1], [([2, 3], [dim + 2, dim + 3])])
    #     f_m1_m2 = (f_m1_m2 + 1).compute()

    #     m1_m2 = np.zeros((dim + 2, dim + 3))
    #     m1_m2[2 : dim + 2, 3 : dim + 3] = m1

    #     m1_m2 += 1
    #     res = np.allclose(f_m1_m2, m1_m2)

    #     if not res:
    #         print("Federated:")
    #         print(f_m1_m2)
    #         print("numpy:")
    #         print(m1_m2)
    #     self.assertTrue(res)

    # def test_10(self):
    #     #   [[m1,m1,m1,m1,m1, 0, 0, 0]
    #     #    [m1,m1,m1,m1,m1, 0, 0, 0]
    #     #    [m1,m1,m1,m1,m1, 0, 0, 0]
    #     #    [m1,m1,m1,m1,m1, 0, 0, 0]
    #     #    [m1,m1,m1,m1,m1, 0, 0, 0]
    #     #    [ 0, 0, 0, 0, 0, 0, 0, 0]
    #     #    [ 0, 0, 0, 0, 0, 0, 0, 0]]
    #     f_m1_m2 = self.sds.federated( [fed1], [([0, 0], [dim + 2, dim + 3])])
    #     f_m1_m2 = (f_m1_m2).compute()

    #     m1_m2 = np.zeros((dim + 2, dim + 3))
    #     m1_m2[0:dim, 0:dim] = m1

    #     res = np.allclose(f_m1_m2, m1_m2)

    #     if not res:
    #         print("Federated:")
    #         print(f_m1_m2)
    #         print("numpy:")
    #         print(m1_m2)
    #     self.assertTrue(res)

    # def test_11(self):
    #     #   [[ 0, 0, 0, 0, 0, 0, 0, 0]
    #     #    [ 0,m1,m1,m1,m1,m1, 0, 0]
    #     #    [ 0,m1,m1,m1,m1,m1, 0, 0]
    #     #    [ 0,m1,m1,m1,m1,m1, 0, 0]
    #     #    [ 0,m1,m1,m1,m1,m1, 0, 0]
    #     #    [ 0,m1,m1,m1,m1,m1, 0, 0]
    #     #    [ 0, 0, 0, 0, 0, 0, 0, 0]]
    #     f_m1_m2 = self.sds.federated( [fed1], [([1, 1], [dim + 2, dim + 3])])
    #     f_m1_m2 = (f_m1_m2).compute()

    #     m1_m2 = np.zeros((dim + 2, dim + 3))
    #     m1_m2[1 : dim + 1, 1 : dim + 1] = m1

    #     res = np.allclose(f_m1_m2, m1_m2)

    #     if not res:
    #         print("Federated:")
    #         print(f_m1_m2)
    #         print("numpy:")
    #         print(m1_m2)
    #     self.assertTrue(res)


if __name__ == "__main__":
    unittest.main(exit=False)
