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

# Seed the randomness.
np.random.seed(7)

m = np.random.rand(3, 4)


class TestFederatedAggFn(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_setup(self):
        sm = self.sds.from_numpy(m)
        sr = sm.compute()
        e = m
        self.assertTrue(np.allclose(e, sr))

    def test_slice_first_third_row(self):
        sm = self.sds.from_numpy(m)[[0, 2]]
        sr = sm.compute()
        e = m[[0, 2]]
        self.assertTrue(np.allclose(e, sr))

    def test_slice_single_row(self):
        sm = self.sds.from_numpy(m)[[1]]
        sr = sm.compute()
        e = m[[1]]
        self.assertTrue(np.allclose(e, sr))

    def test_slice_last_row(self):
        with self.assertRaises(ValueError):
            self.sds.from_numpy(m)[[-1]]

    def test_slice_first_third_col(self):
        sm = self.sds.from_numpy(m)[:, [0, 2]]
        sr = sm.compute()
        e = m[:, [0, 2]]
        self.assertTrue(np.allclose(e, sr))

    def test_slice_single_col(self):
        sm = self.sds.from_numpy(m)[:, [1]]
        sr = sm.compute()
        e = m[:, [1]]
        self.assertTrue(np.allclose(e, sr))

    def test_slice_row_col_both(self):
        with self.assertRaises(NotImplementedError):
            self.sds.from_numpy(m)[[1, 2], [0, 3]]


if __name__ == "__main__":
    unittest.main(exit=False)
