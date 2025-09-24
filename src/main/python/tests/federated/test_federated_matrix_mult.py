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
import time
import unittest

import numpy as np
from systemds.context import SystemDSContext

os.environ["SYSDS_QUIET"] = "1"

dim = 3

m = np.reshape(np.arange(1, dim * dim + 1, 1), (dim, dim))
m_c2 = np.column_stack((m, m))
m_c3 = np.column_stack((m, m_c2))
m_r2 = np.row_stack((m, m))
m_r3 = np.row_stack((m, m_r2))

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
fed_c2_file = tempdir + "m_c2.fed"
fed_c3_file = tempdir + "m_c3.fed"
fed_r2_file = tempdir + "m_r2.fed"
fed_r3_file = tempdir + "m_r3.fed"


class TestFederatedAggFn(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)
        cls.sds.federated([fed1], [([0, 0], [dim, dim])]).write(
            fed1_file, format="federated"
        ).compute()
        cls.sds.federated(
            [fed1, fed2], [([0, 0], [dim, dim]), ([0, dim], [dim, dim * 2])]
        ).write(fed_c2_file, format="federated").compute()
        cls.sds.federated(
            [fed1, fed2, fed3],
            [
                ([0, 0], [dim, dim]),
                ([0, dim], [dim, dim * 2]),
                ([0, dim * 2], [dim, dim * 3]),
            ],
        ).write(fed_c3_file, format="federated").compute()
        cls.sds.federated(
            [fed1, fed2], [([0, 0], [dim, dim]), ([dim, 0], [dim * 2, dim])]
        ).write(fed_r2_file, format="federated").compute()
        cls.sds.federated(
            [fed1, fed2, fed3],
            [
                ([0, 0], [dim, dim]),
                ([dim, 0], [dim * 2, dim]),
                ([dim * 2, 0], [dim * 3, dim]),
            ],
        ).write(fed_r3_file, format="federated").compute()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    #####################
    # Single site tests #
    #####################

    def test_single_fed_site_same_matrix(self):
        f_m = self.sds.read(fed1_file)
        self.exec_test(m, m, f_m, f_m)

    def test_single_fed_left_same_size(self):
        f_m = self.sds.read(fed1_file)
        m_s = self.sds.from_numpy(m)
        self.exec_test(m, m, m_s, f_m)

    def test_single_fed_left_plus_one_row(self):
        f_m = self.sds.read(fed1_file)
        m_row_plus1 = np.reshape(np.arange(1, dim * (dim + 1) + 1, 1), (dim + 1, dim))
        m_s = self.sds.from_numpy(m_row_plus1)
        self.exec_test(m_row_plus1, m, m_s, f_m)

    def test_single_fed_left_minus_one_row(self):
        f_m = self.sds.read(fed1_file)
        m_row_minus1 = np.reshape(np.arange(1, dim * (dim - 1) + 1, 1), (dim - 1, dim))
        m_s = self.sds.from_numpy(m_row_minus1)
        self.exec_test(m_row_minus1, m, m_s, f_m)

    def test_single_fed_left_vector_row(self):
        f_m = self.sds.read(fed1_file)
        v_row = np.arange(1, dim + 1, 1)
        v_s = self.sds.from_numpy(v_row).t()
        self.exec_test(v_row, m, v_s, f_m)

    def test_single_fed_right_same_size(self):
        f_m = self.sds.read(fed1_file)
        m_s = self.sds.from_numpy(m)
        self.exec_test(m, m, f_m, m_s)

    def test_single_fed_right_plus_one_row(self):
        f_m = self.sds.read(fed1_file)
        m_col_plus1 = np.reshape(np.arange(1, dim * (dim + 1) + 1, 1), (dim, dim + 1))
        m_s = self.sds.from_numpy(m_col_plus1)
        self.exec_test(m, m_col_plus1, f_m, m_s)

    def test_single_fed_right_minus_one_row(self):
        f_m = self.sds.read(fed1_file)
        m_col_minus1 = np.reshape(np.arange(1, dim * (dim - 1) + 1, 1), (dim, dim - 1))
        m_s = self.sds.from_numpy(m_col_minus1)
        self.exec_test(m, m_col_minus1, f_m, m_s)

    def test_single_fed_right_vector(self):
        f_m = self.sds.read(fed1_file)
        v_col = np.reshape(np.arange(1, dim + 1, 1), (1, dim))
        v_col_sds = self.sds.from_numpy(v_col).t()
        self.exec_test(m, np.transpose(v_col), f_m, v_col_sds)

    ##################################
    # start two federated site tests #
    ##################################

    def test_two_fed_standard(self):
        f_m2 = self.sds.read(fed_c2_file)
        m = np.reshape(np.arange(1, dim * (dim + dim) + 1, 1), (dim * 2, dim))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m, m_c2, m_s, f_m2)

    def test_two_fed_left_minus_one_row(self):
        f_m2 = self.sds.read(fed_c2_file)
        m = np.reshape(np.arange(1, dim * (dim + dim - 1) + 1, 1), (dim * 2 - 1, dim))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m, m_c2, m_s, f_m2)

    def test_two_fed_left_plus_one_row(self):
        f_m2 = self.sds.read(fed_c2_file)
        m = np.reshape(np.arange(1, dim * (dim + dim + 1) + 1, 1), (dim * 2 + 1, dim))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m, m_c2, m_s, f_m2)

    def test_two_fed_left_vector_row(self):
        f_m2 = self.sds.read(fed_c2_file)
        m = np.arange(1, dim + 1, 1)
        m_s = self.sds.from_numpy(m).t()
        self.exec_test(m, m_c2, m_s, f_m2)

    def test_two_fed_right_standard(self):
        f_m2 = self.sds.read(fed_c2_file)
        m_s = self.sds.from_numpy(m_r2)
        self.exec_test(m_c2, m_r2, f_m2, m_s)

    def test_two_fed_right_col_minus_1(self):
        f_m2 = self.sds.read(fed_c2_file)
        m = np.reshape(np.arange(1, (dim - 1) * (dim + dim) + 1, 1), (dim * 2, dim - 1))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m_c2, m, f_m2, m_s)

    def test_two_fed_right_col_plus_1(self):
        f_m2 = self.sds.read(fed_c2_file)
        m = np.reshape(np.arange(1, (dim + 1) * (dim + dim) + 1, 1), (dim * 2, dim + 1))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m_c2, m, f_m2, m_s)

    def test_two_fed_right_vector(self):
        f_m2 = self.sds.read(fed_c2_file)
        m = np.reshape(np.arange(1, (dim + dim) + 1, 1), (dim * 2, 1))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m_c2, m, f_m2, m_s)

    ####################################
    # Start three federated site tests #
    ####################################

    def test_three_fed_standard(self):
        f_m3 = self.sds.read(fed_c3_file)
        m = np.reshape(np.arange(1, dim * (dim * 3) + 1, 1), (dim * 3, dim))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m, m_c3, m_s, f_m3)

    def test_three_fed_left_minus_one_row(self):
        f_m3 = self.sds.read(fed_c3_file)
        m = np.reshape(np.arange(1, dim * (dim * 3 - 1) + 1, 1), (dim * 3 - 1, dim))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m, m_c3, m_s, f_m3)

    def test_three_fed_left_plus_one_row(self):
        f_m3 = self.sds.read(fed_c3_file)
        m = np.reshape(np.arange(1, dim * (dim * 3 + 1) + 1, 1), (dim * 3 + 1, dim))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m, m_c3, m_s, f_m3)

    def test_three_fed_left_vector_row(self):
        f_m3 = self.sds.read(fed_c3_file)
        m = np.arange(1, dim + 1, 1)
        m_s = self.sds.from_numpy(m).t()
        self.exec_test(m, m_c3, m_s, f_m3)

    def test_three_fed_right_standard(self):
        f_m3 = self.sds.read(fed_c3_file)
        m_s = self.sds.from_numpy(m_r3)
        self.exec_test(m_c3, m_r3, f_m3, m_s)

    def test_three_fed_right_col_minus_1(self):
        f_m3 = self.sds.read(fed_c3_file)
        m = np.reshape(np.arange(1, (dim - 1) * (dim * 3) + 1, 1), (dim * 3, dim - 1))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m_c3, m, f_m3, m_s)

    def test_three_fed_right_col_plus_1(self):
        f_m3 = self.sds.read(fed_c3_file)
        m = np.reshape(np.arange(1, (dim + 1) * (dim * 3) + 1, 1), (dim * 3, dim + 1))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m_c3, m, f_m3, m_s)

    def test_three_fed_right_vector(self):
        f_m3 = self.sds.read(fed_c3_file)
        m = np.reshape(np.arange(1, (dim * 3) + 1, 1), (dim * 3, 1))
        m_s = self.sds.from_numpy(m)
        self.exec_test(m_c3, m, f_m3, m_s)

    ###################
    # row bind matrix #
    ###################

    def test_federated_row2_binded(self):
        fed = self.sds.read(fed_r2_file)
        s_m = self.sds.from_numpy(m_c2)
        self.exec_test(m_c2, m_r2, s_m, fed)

    def test_federated_row3_binded(self):
        fed = self.sds.read(fed_r3_file)
        s_m = self.sds.from_numpy(m_c3)
        self.exec_test(m_c3, m_r3, s_m, fed)

    def test_previously_failing(self):
        # local matrix to multiply with
        loc = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
            ]
        )
        # Multiply local and federated
        ret_loc = loc @ m_r3

        for i in range(1, 100):
            loc_systemds = self.sds.from_numpy(loc)
            fed = self.sds.read(fed_r3_file)
            ret_fed = (loc_systemds @ fed).compute()
            if not np.allclose(ret_fed, ret_loc):
                self.fail("not equal outputs of federated matrix multiplications")

    def exec_test(self, left, right, f_left, f_right):
        fed = f_left @ f_right
        loc = left @ right
        fed_res = fed.compute()
        self.assertTrue(np.allclose(fed_res, loc))


if __name__ == "__main__":
    unittest.main(exit=False)
