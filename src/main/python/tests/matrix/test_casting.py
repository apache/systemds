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
from pandas import DataFrame
from numpy import ndarray


class TestDIAG(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    def tearDown(self):
        self.sds.close()

    def test_casting_basic1(self):
        sds_input = self.sds.from_numpy(np.array([[1]]))
        sds_result = sds_input.to_scalar().compute()
        self.assertTrue(type(sds_result) == float)

    def test_casting_basic2(self):
        sds_input = self.sds.from_numpy(np.array([[1]]))
        sds_result = sds_input.to_frame().compute()
        self.assertTrue(type(sds_result) == DataFrame)

    def test_casting_basic3(self):
        sds_result = self.sds.scalar(1.0).to_frame().compute()
        self.assertTrue(type(sds_result) == DataFrame)

    def test_casting_basic4(self):
        sds_result = self.sds.scalar(1.0).to_matrix().compute()
        self.assertTrue(type(sds_result) == ndarray)

    def test_casting_basic5(self):
        ar = ndarray((2, 2))
        df = DataFrame(ar)
        sds_result = self.sds.from_pandas(df).to_matrix().compute()
        self.assertTrue(type(sds_result) == ndarray and np.allclose(ar, sds_result))

    def test_casting_basic6(self):
        ar = ndarray((1, 1))
        df = DataFrame(ar)
        sds_result = self.sds.from_pandas(df).to_scalar().compute()
        self.assertTrue(type(sds_result) == float)

    def test_casting_basic7(self):
        sds_result = self.sds.scalar(1.0).to_int().compute()
        self.assertTrue(type(sds_result) == int and sds_result)

    def test_casting_basic8(self):
        sds_result = self.sds.scalar(1.0).to_boolean().compute()
        self.assertTrue(type(sds_result) == bool)


if __name__ == "__main__":
    unittest.main()
