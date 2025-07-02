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

from systemds.context import SystemDSContext


class Test__str__(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_1(self):
        self.assertTrue("MatrixNode" in str(self.sds.full([1, 2], 3)))

    def test_2(self):
        self.assertTrue("ScalarNode" in str(self.sds.scalar(3)))

    def test_3(self):
        self.assertTrue("ScalarNode" in str(self.sds.scalar("Hi")))

    def test_4(self):
        self.assertTrue("ScalarNode" in str(self.sds.full([1, 2], 3).to_string()))

    def test_5(self):
        self.assertTrue(
            "ListNode"
            in str(self.sds.list(self.sds.rand(1, 2, 3, 4), self.sds.scalar(4)))
        )

    def test_6(self):
        self.assertTrue(
            "MatrixNode"
            in str(self.sds.list(self.sds.rand(1, 2, 3, 4), self.sds.scalar(4))[0])
        )

    def test_7(self):
        self.assertTrue(
            "ScalarNode"
            in str(self.sds.list(self.sds.rand(1, 2, 3, 4), self.sds.scalar(4))[1])
        )


if __name__ == "__main__":
    unittest.main(exit=False)
