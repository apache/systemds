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


class TestSource_MultiArguments(unittest.TestCase):
    sds: SystemDSContext = None
    src_path: str = "./tests/source/source_multi_arguments.dml"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_01(self):
        s = self.sds.source(self.src_path, "test")

        m1 = self.sds.rand(12, 1)
        m2 = self.sds.rand(1, 2)
        m3 = self.sds.rand(23, 3)
        c = s.blaaa_is_a_BAAD_function_name_but_it_works(m1, m2, m3)

        res = c.compute()
        self.assertEqual(1, self.imports(c.script_str))

    # def test_02(self):
    #     s = self.sds.source(self.src_path,"test")

    #     m = self.sds.rand(12,1)
    #     c = s.blaaa_is_a_BAAD_function_name_but_it_works(m,m,m)

    #     res = c.compute()
    #     self.assertEqual(1, self.imports(c.script_str))
    #     self.assertTrue("V3" not in c.script_str, "Only 2 variables should be allocated.")

    def imports(self, script: str) -> int:
        return script.split("\n").count(f'source("{self.src_path}") as test')


if __name__ == "__main__":
    unittest.main(exit=False)
