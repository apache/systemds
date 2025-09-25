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

dim = 5
m1 = np.array(np.random.randint(100, size=dim * dim) + 1.01, dtype=np.double)
m1.shape = (dim, dim)
m2 = np.random.choice(np.arange(0.01, 1, 0.1), size=(dim, dim))
s = 3.02


class TestTrigonometricOp(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_sin(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).sin().compute(), np.sin(m1))
        )

    def test_cos(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).cos().compute(), np.cos(m1))
        )

    def test_tan(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).tan().compute(), np.tan(m1))
        )

    def test_asin(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m2).asin().compute(), np.arcsin(m2))
        )

    def test_acos(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m2).acos().compute(), np.arccos(m2))
        )

    def test_atan(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m2).atan().compute(), np.arctan(m2))
        )

    def test_sinh(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).sinh().compute(), np.sinh(m1))
        )

    def test_cosh(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).cosh().compute(), np.cosh(m1))
        )

    def test_tanh(self):
        self.assertTrue(
            np.allclose(self.sds.from_numpy(m1).tanh().compute(), np.tanh(m1))
        )


if __name__ == "__main__":
    unittest.main(exit=False)
