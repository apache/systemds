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
from time import sleep
from systemds.context import SystemDSContext


class TestPrint(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True)
        sleep(2.0)
        # Clear stdout ...
        cls.sds.get_stdout()
        cls.sds.get_stdout()
        sleep(1.0)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_print_01(self):
        self.sds.from_numpy(np.array([1])).to_string().print().compute()
        sleep(0.2)
        self.assertEqual(1, float(self.sds.get_stdout()[0].replace(",", ".")))

    def test_print_02(self):
        self.sds.scalar(1).print().compute()
        sleep(0.2)
        self.assertEqual(1, float(self.sds.get_stdout()[0]))


if __name__ == "__main__":
    unittest.main(exit=False)
