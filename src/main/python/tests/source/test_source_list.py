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
from systemds.operator.algorithm.builtin.scale import scale


class TestSource_01(unittest.TestCase):
    sds: SystemDSContext = None
    source_path: str = "./tests/source/source_with_list_input.dml"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_single_return(self):
        arr = self.sds.array(self.sds.full((10, 10), 4))
        c = self.sds.source(self.source_path, "test").func(arr)
        res = c.sum().compute()
        self.assertTrue(res == 10 * 10 * 4)

    def test_input_multireturn(self):
        m = self.sds.full((10, 10), 2)
        [a, b, c] = scale(m, center=True, scale=True)
        arr = self.sds.array(a, b, c)
        c = self.sds.source(self.source_path, "test").func(arr)
        res = c.sum().compute()
        self.assertTrue(res == 0)

    # [SYSTEMDS-3224] https://issues.apache.org/jira/browse/SYSTEMDS-3224
    # def test_multi_return(self):
    #     arr = self.sds.array(
    #         self.sds.full((10, 10), 4),
    #         self.sds.full((3, 3), 5))
    #     [b, c] = self.sds.source(self.source_path, "test", True).func2(arr)
    #     res = c.sum().compute()
    #     self.assertTrue(res == 10*10*4)


if __name__ == "__main__":
    unittest.main(exit=False)
