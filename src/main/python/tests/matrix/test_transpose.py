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
import random

import numpy as np
from systemds.context import SystemDSContext

np.random.seed(7)

shape = (random.randrange(1, 25), random.randrange(1, 25))
m = np.random.rand(shape[0], shape[1])
mx = np.random.rand(1, shape[1])
my = np.random.rand(shape[0], 1)


class TestTranspose(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_basic(self):
        trans = self.sds.from_numpy(m).t().compute()
        self.assertTrue(np.allclose(trans, np.transpose(m)))

    def test_empty(self):
        trans = self.sds.from_numpy(np.asarray([])).t().compute()
        self.assertTrue(np.allclose(trans, np.asarray([])))

    def test_row(self):
        trans = self.sds.from_numpy(mx).t().compute()
        self.assertTrue(np.allclose(trans, np.transpose(mx)))

    def test_col(self):
        trans = self.sds.from_numpy(my).t().compute()
        self.assertTrue(np.allclose(trans, np.transpose(my)))


if __name__ == "__main__":
    unittest.main(exit=False)
