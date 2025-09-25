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

    def test_01_single_call(self):
        c = self.sds.source("./tests/source/source_01.dml", "test").test_01()
        res = c.compute()
        self.assertEqual(1, self.imports(c.script_str))
        self.assertTrue(np.allclose(np.array([[1]]), res))

    def test_01_multi_call_01(self):
        s = self.sds.source("./tests/source/source_01.dml", "test")
        a = s.test_01()
        b = s.test_01()
        c = a + b
        res = c.compute()
        self.assertEqual(1, self.imports(c.script_str))
        self.assertTrue(np.allclose(np.array([[2]]), res))

    def test_01_multi_call_02(self):
        s = self.sds.source("./tests/source/source_01.dml", "test")
        a = s.test_01()
        b = s.test_01()
        c = a + b + a
        res = c.compute()
        self.assertEqual(1, self.imports(c.script_str))
        self.assertTrue(np.allclose(np.array([[3]]), res))

    def test_01_invalid_function(self):
        s = self.sds.source("./tests/source/source_01.dml", "test")
        with self.assertRaises(AttributeError) as context:
            a = s.test_01_NOT_A_REAL_FUNCTION()

    def test_01_invalid_arguments(self):
        s = self.sds.source("./tests/source/source_01.dml", "test")
        m = self.sds.full((1, 1), 2)
        with self.assertRaises(TypeError) as context:
            a = s.test_01(m)

    def test_01_sum(self):
        c = self.sds.source("./tests/source/source_01.dml", "test").test_01().sum()
        res = c.compute()
        self.assertEqual(1, self.imports(c.script_str))
        self.assertTrue(np.allclose(np.array([[1]]), res))

    def imports(self, script: str) -> int:
        return script.split("\n").count(
            'source("./tests/source/source_01.dml") as test'
        )


if __name__ == "__main__":
    unittest.main(exit=False)
