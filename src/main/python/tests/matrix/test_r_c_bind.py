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


class TestRBind(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_r_bind(self):
        m1 = self.sds.from_numpy(np.zeros((10, 1)))
        m2 = self.sds.from_numpy(np.ones((10, 1)))
        res = m1.rbind(m2).compute()
        npres = np.vstack((np.zeros((10, 1)), np.ones((10, 1))))
        self.assertTrue(np.allclose(res, npres))

    def test_c_bind(self):
        m1 = self.sds.from_numpy(np.zeros((10, 6)))
        m2 = self.sds.from_numpy(np.ones((10, 7)))
        res = m1.cbind(m2).compute()
        npres = np.hstack((np.zeros((10, 6)), np.ones((10, 7))))
        self.assertTrue(np.allclose(res, npres))


if __name__ == "__main__":
    unittest.main(exit=False)
