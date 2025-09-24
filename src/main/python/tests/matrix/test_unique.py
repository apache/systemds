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


np.random.seed(7)


# np's unique applied on an axis checks for unique vectors along that axis -> on the other hand systemds' unique
# returns the unique values along that axis for each vector on that axis
def compute_expected(m, num_cols, axis):
    def padded(row):
        unique = np.unique(row)
        row = np.pad(unique, (num_cols - len(unique), 0), "constant", constant_values=0)
        return row

    if axis == 1:
        return np.array([padded(r) for r in m])
    else:
        return np.array([padded(r) for r in m.T]).T


class TestUNIQUE(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    def tearDown(self):
        self.sds.close()

    def test_unique_basic(self):
        input_matrix = np.array(
            [[1, -2, 3, 4], [0, -6, 7, 8], [0, -10, 11, -12], [0, -14, 15, -16]]
        )

        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.unique().compute()
        sds_result = np.sort(np.reshape(sds_result, (-1)))
        np_result = np.unique(input_matrix)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_unique_basic2(self):
        input_matrix = np.array(
            [[1, 1, 1, 1], [2, 2, 2, 2], [0, 10, 11, 12], [0, 14, 15, 16]]
        )

        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.unique(1).compute()
        sds_result = np.sort(sds_result, 1)
        num_cols = sds_result.shape[1]
        np_result = compute_expected(input_matrix, num_cols, 1)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_unique_basic3(self):
        input_matrix = np.array(
            [[0, 1, 1, 1], [0, 1, 1, 1], [0, 10, 11, 12], [0, 14, 15, 16]]
        )

        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.unique(0).compute()
        sds_result = np.sort(sds_result, 0)
        num_rows = sds_result.shape[0]
        np_result = compute_expected(input_matrix, num_rows, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_unique_random1(self):
        input_matrix = np.random.random((10, 10)) * 200
        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.unique().compute()
        sds_result = np.sort(np.reshape(sds_result, (-1)))
        np_result = np.unique(input_matrix)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_unique_random2(self):
        input_matrix = np.random.random((10, 10)) * 200
        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.unique(1).compute()
        sds_result = np.sort(sds_result, 1)
        num_cols = sds_result.shape[1]
        np_result = compute_expected(input_matrix, num_cols, 1)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_unique_random3(self):
        input_matrix = np.random.random((10, 10)) * 200
        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.unique(0).compute()
        sds_result = np.sort(sds_result, 0)
        num_rows = sds_result.shape[0]
        np_result = compute_expected(input_matrix, num_rows, 0)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_unique_error(self):
        with self.assertRaises(ValueError):
            self.sds.from_numpy(np.array([[1, 2]])).unique(2)


if __name__ == "__main__":
    unittest.main()
