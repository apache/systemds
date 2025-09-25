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


class TestSource_01(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_func_01(self):
        c = self.sds.source("./tests/source/source_02.dml", "test").func_01()
        res = c.compute()
        self.assertEqual(1, self.imports(c.script_str))
        self.assertTrue(np.allclose(np.array([[1]]), res))

    def test_func_02(self):
        m = self.sds.full((3, 5), 2)
        c = self.sds.source("./tests/source/source_02.dml", "test").func_02(m)
        res = c.compute()
        self.assertEqual(1, self.imports(c.script_str))
        self.assertEqual(1, res.shape[1])

    def test_func_02_call_self(self):
        m = self.sds.full((3, 2), 2)
        s = self.sds.source("./tests/source/source_02.dml", "test")
        c = s.func_02(m)
        cc = s.func_02(c)
        res = cc.compute()
        self.assertEqual(1, self.imports(cc.script_str))
        self.assertEqual(1, res.shape[1])

    def test_func_02_sum(self):
        m = self.sds.full((3, 5), 2)
        c = self.sds.source("./tests/source/source_02.dml", "test").func_02(m)
        c = c.sum()
        res = c.compute()
        self.assertEqual(1, self.imports(c.script_str))

    def test_Preprocess_sum(self):
        m = self.sds.full((3, 5), 2)
        c = self.sds.source("./tests/source/source_02.dml", "test").Preprocess(m)
        c = c.sum()
        res = c.compute()
        self.assertEqual(1, self.imports(c.script_str))

    def imports(self, script: str) -> int:
        return script.split("\n").count(
            'source("./tests/source/source_02.dml") as test'
        )


if __name__ == "__main__":
    unittest.main(exit=False)
