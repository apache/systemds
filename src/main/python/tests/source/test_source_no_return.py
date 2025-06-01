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

from time import sleep
from systemds.context import SystemDSContext


class TestSource_NoReturn(unittest.TestCase):
    sds: SystemDSContext = None
    src_path: str = "./tests/source/source_with_no_return.dml"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_01(self):
        s = self.sds.source(self.src_path, "test")
        c = s.no_return()
        c.compute()
        sleep(1)  # to allow the std buffer to fill
        stdout = self.sds.get_stdout()
        self.assertEqual(4.2 + 14 * 2, float(stdout[0]))

    def test_02(self):
        s = self.sds.source(self.src_path, "test")
        c = s.no_return(4)
        c.compute()
        sleep(1)  # to allow the std buffer to fill
        stdout = self.sds.get_stdout()
        self.assertEqual(4 + 14 * 2, float(stdout[0]))

    def test_03(self):
        s = self.sds.source(self.src_path, "test")
        c = s.no_return(a=14)
        c.compute()
        sleep(1)  # to allow the std buffer to fill
        stdout = self.sds.get_stdout()
        self.assertEqual(14 + 14 * 2, float(stdout[0]))


if __name__ == "__main__":
    unittest.main(exit=False)
