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


class TestSourceReuse(unittest.TestCase):

    sds: SystemDSContext = None
    source_reuse = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)
        cls.source_reuse = cls.sds.source("./tests/source/source_01.dml", "test")

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_01_single_call(self):
        self.call()

    def test_02_second_call(self):
        self.call()

    def test_03_same_function(self):
        s = self.sds.source("./tests/source/source_01.dml", "test")
        c = s.test_01().compute()
        d = s.test_01().compute()
        self.assertTrue(np.allclose(c, d))

    def call(self):
        c = self.source_reuse.test_01()
        res = c.compute()
        self.assertEqual(1, self.imports(c.script_str))
        self.assertTrue(np.allclose(np.array([[1]]), res))

    def imports(self, script: str) -> int:
        return script.split("\n").count(
            'source("./tests/source/source_01.dml") as test'
        )


if __name__ == "__main__":
    unittest.main(exit=False)
