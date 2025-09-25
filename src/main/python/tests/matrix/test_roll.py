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
import random
from scipy import sparse
from systemds.context import SystemDSContext

np.random.seed(7)
random.seed(7)
shape = (random.randrange(1, 25), random.randrange(1, 25))

m = np.random.rand(shape[0], shape[1])
my = np.random.rand(shape[0], 1)
m_empty = np.asarray([[]])
m_sparse = sparse.random(
    shape[0], shape[1], density=0.1, format="csr", random_state=5
).toarray()
m_sparse = np.around(m_sparse, decimals=22)


class TestRoll(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_empty(self):
        r = self.sds.from_numpy(np.asarray(m_empty)).roll(1).compute()
        self.assertTrue(np.allclose(r, m_empty))

    def test_col_vec(self):
        r = self.sds.from_numpy(my).roll(1).compute()
        self.assertTrue(np.allclose(r, np.roll(my, axis=None, shift=1)))

    def test_basic(self):
        r = self.sds.from_numpy(m).roll(1).compute()
        self.assertTrue(np.allclose(r, np.roll(m, axis=0, shift=1)))

    def test_sparse_matrix(self):
        r = self.sds.from_numpy(m_sparse).roll(1).compute()
        self.assertTrue(np.allclose(r, np.roll(m_sparse, axis=0, shift=1)))


if __name__ == "__main__":
    unittest.main(exit=False)
