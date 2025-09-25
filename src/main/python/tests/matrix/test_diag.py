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


class TestDIAG(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    def tearDown(self):
        self.sds.close()

    def test_diag_basic1(self):
        input_matrix = np.array([1, 2, 3, 4])

        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.diag().compute()
        np_result = np.diag(input_matrix)
        print(np_result)
        print(sds_result)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_diag_basic2(self):
        input_matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )

        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.diag().compute()
        np_result = np.reshape(np.diag(input_matrix), (-1, 1))
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_diag_random1(self):
        input_matrix = np.random.random(10)
        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.diag().compute()
        np_result = np.diag(input_matrix)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_diag_random2(self):
        input_matrix = np.random.random((10, 10))
        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.diag().compute()
        np_result = np.reshape(np.diag(input_matrix), (-1, 1))
        assert np.allclose(sds_result, np_result, 1e-9)


if __name__ == "__main__":
    unittest.main()
