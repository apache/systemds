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
import time

from systemds.context import SystemDSContext
from systemds.script_building import DMLScript


class Test_DMLScript(unittest.TestCase):
    """Test class for testing behavior of the fundamental DMLScript class
    """

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_simple_print_1(self):
        script = DMLScript(self.sds)
        script.add_code('print("Hello")')
        script.execute()
        time.sleep(0.5)
        stdout = self.sds.get_stdout(100)
        self.assertListEqual(["Hello"], stdout)

    def test_simple_print_2(self):
        script = DMLScript(self.sds)
        script.add_code('print("Hello")')
        script.add_code('print("World")')
        script.add_code('print("!")')
        script.execute()
        time.sleep(0.5)
        stdout = self.sds.get_stdout(100)
        self.assertListEqual(['Hello', 'World', '!'], stdout)

    def test_multiple_executions_1(self):
        scr_a = DMLScript(self.sds)
        scr_a.add_code('x = 4')
        scr_a.add_code('print(x)')
        scr_a.add_code('y = x + 1')
        scr_a.add_code('print(y)')
        scr_a.execute()
        time.sleep(0.5)
        stdout = self.sds.get_stdout(100)
        self.assertEqual("4", stdout[0])
        self.assertEqual("5", stdout[1])


if __name__ == "__main__":
    unittest.main(exit=False)
