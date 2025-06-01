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

np.random.seed(1412)


class TestContextCreation(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def getM(self):
        m1 = np.array(np.random.randint(10, size=5 * 5), dtype=np.int64)
        m1.shape = (5, 5)
        return m1

    def test_stats_v1(self):
        a = self.sds.from_numpy(self.getM())
        a = a + 1
        a = a * 4
        a = a + 3
        a = a / 23

        self.sds.capture_stats()
        a.compute()
        self.sds.capture_stats(False)

        stats = self.sds.get_stats()
        self.sds.clear_stats()
        instructions = "\n".join(
            stats.split("Heavy hitter instructions:")[1].split("\n")[2:]
        )
        assert "+" in instructions and "*" in instructions and "/" in instructions


if __name__ == "__main__":
    unittest.main(exit=False)
