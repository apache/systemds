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

m1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

m2 = np.random.random((10, 10))


class TestTRIANGULAR(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    def tearDown(self):
        self.sds.close()

    def test_triu_basic1(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.triu().compute()
        np_result = np.triu(m1)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_triu_basic2(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.triu(include_diagonal=False).compute()
        np_result = np.triu(m1, 1)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_triu_basic3(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.triu(return_values=False).compute()
        np_result = np.triu(m1) > 0
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_triu_basic4(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.triu(
            return_values=False, include_diagonal=False
        ).compute()
        np_result = np.triu(m1, 1) > 0
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_triu_random(self):
        sds_input = self.sds.from_numpy(m2)
        sds_result = sds_input.triu().compute()
        np_result = np.triu(m2)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_tril_basic1(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.tril().compute()
        np_result = np.tril(m1)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_tril_basic2(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.tril(include_diagonal=False).compute()
        np_result = np.tril(m1, -1)
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_tril_basic3(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.tril(return_values=False).compute()
        np_result = np.tril(m1) > 0
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_tril_basic4(self):
        sds_input = self.sds.from_numpy(m1)
        sds_result = sds_input.tril(
            return_values=False, include_diagonal=False
        ).compute()
        np_result = np.tril(m1, -1) > 0
        assert np.allclose(sds_result, np_result, 1e-9)

    def test_tril_random(self):
        sds_input = self.sds.from_numpy(m2)
        sds_result = sds_input.tril().compute()
        np_result = np.tril(m2)
        assert np.allclose(sds_result, np_result, 1e-9)


if __name__ == "__main__":
    unittest.main()
