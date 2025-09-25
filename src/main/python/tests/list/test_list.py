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
from systemds.operator.algorithm import pca


class TestListOperations(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_creation(self):
        """
        Tests the creation of a List object via the SystemDSContext
        """
        m1 = np.array([1.0, 2.0, 3.0])
        m1p = self.sds.from_numpy(m1)
        m2 = np.array([4.0, 5.0, 6.0])
        m2p = self.sds.from_numpy(m2)
        list_obj = self.sds.array(m1p, m2p)
        tmp = list_obj[0] + list_obj[1]
        res = tmp.compute().flatten()
        self.assertTrue(np.allclose(m1 + m2, res))

    def test_addition(self):
        """
        Tests the creation of a List object via the SystemDSContext and adds a value
        """
        m1 = np.array([1.0, 2.0, 3.0])
        m1p = self.sds.from_numpy(m1)
        m2 = np.array([4.0, 5.0, 6.0])
        m2p = self.sds.from_numpy(m2)
        list_obj = self.sds.array(m1p, m2p)
        tmp = list_obj[0] + 2
        res = tmp.compute().flatten()
        self.assertTrue(np.allclose(m1 + 2, res))


if __name__ == "__main__":
    unittest.main(exit=False)
