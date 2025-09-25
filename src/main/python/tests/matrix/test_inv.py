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
m = np.random.random((10, 10))


class TestINV(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    def tearDown(self):
        self.sds.close()

    def test_diag_basic(self):
        input_matrix = np.array([[2, 0], [0, 6]])
        sds_input = self.sds.from_numpy(input_matrix)
        sds_result = sds_input.inv().compute()
        np_result = np.linalg.inv(input_matrix)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_diag_random(self):
        sds_input = self.sds.from_numpy(m)
        sds_result = sds_input.inv().compute()
        np_result = np.linalg.inv(m)
        assert np.allclose(sds_result, np_result, 1e-9)


if __name__ == "__main__":
    unittest.main()
