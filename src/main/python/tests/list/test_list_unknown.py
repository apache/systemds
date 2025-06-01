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
from systemds.operator import List
from systemds.operator.algorithm import pca


class TestListOperationsUnknown(unittest.TestCase):
    sds: SystemDSContext = None
    src_path: str = "./tests/list/return_list.dml"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_access_other_index_1(self):
        s = self.sds.source(self.src_path, "func")
        res = s.f()[1].as_matrix().compute()[0]
        self.assertEqual(1, res)

    def test_access_other_index_2(self):
        s = self.sds.source(self.src_path, "func")
        res = s.f()[2].as_matrix().compute()
        self.assertTrue(np.allclose(np.full((2, 2), 2), res))

    def test_access_other_index_3(self):
        s = self.sds.source(self.src_path, "func")
        res = s.f()[3].as_matrix().compute()
        self.assertTrue(np.allclose(np.full((3, 3), 3), res))


if __name__ == "__main__":
    unittest.main(exit=False)
