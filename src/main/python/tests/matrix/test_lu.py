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
import scipy


class TestLU(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    def tearDown(self):
        self.sds.close()

    def test_lu_error(self):
        # singular matrix
        input_matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )
        with self.assertRaises(RuntimeError):
            self.sds.from_numpy(input_matrix).lu().compute()

    def test_lu_random(self):
        np.random.seed(7)
        input_matrix = np.random.random((10, 10))

        sds_p, sds_l, sds_u = self.sds.from_numpy(input_matrix).lu().compute()
        check = sds_p.dot(input_matrix) - sds_l.dot(sds_u)
        self.assertTrue(np.allclose(check, 0, 1e-9))


if __name__ == "__main__":
    unittest.main()
