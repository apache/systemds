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


class TestSignal(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True, logging_level=50)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_create_signal(self):
        # https://issues.apache.org/jira/browse/SYSTEMDS-3354

        # signal = self.sds.from_numpy(np.arange(0, 3, 1))
        signal = self.sds.seq(0, 2, 1)
        pi = self.sds.scalar(3.141592654)
        size = signal.nRow()
        n = self.sds.seq(0, size - 1)
        k = self.sds.seq(0, size - 1)
        M = (n @ (k.t())) * (2 * pi / size)
        Xa = M.cos() @ signal
        Xb = M.sin() @ signal
        DFT = signal.cbind(Xa).cbind(Xb).compute()


if __name__ == "__main__":
    unittest.main(exit=False)
