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


class TestSource_DefaultValues(unittest.TestCase):
    sds: SystemDSContext = None
    src_path: str = "./tests/source/source_with_default_values.dml"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_01(self):
        s = self.sds.source(self.src_path, "test")
        c = s.d()
        res = c.compute()
        self.assertEqual(4.2, res)

    def test_02(self):
        s = self.sds.source(self.src_path, "test")
        c = s.d(a=self.sds.scalar(5))
        res = c.compute()
        self.assertEqual(5, res)

    def test_03(self):
        s = self.sds.source(self.src_path, "test")
        c = s.d(a=5)
        res = c.compute()
        self.assertEqual(5, res)

    def test_04(self):
        s = self.sds.source(self.src_path, "test")
        c = s.d(c=False)
        res = c.compute()
        self.assertEqual(10, res)

    def test_05(self):
        s = self.sds.source(self.src_path, "test")
        c = s.d(b=1, c=False)
        res = c.compute()
        self.assertEqual(1, res)


if __name__ == "__main__":
    unittest.main(exit=False)
