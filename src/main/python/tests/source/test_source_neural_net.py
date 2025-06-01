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


class TestSource_NeuralNet(unittest.TestCase):
    sds: SystemDSContext = None
    src_path: str = "./tests/source/neural_net_source.dml"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_01(self):
        # Verify that it parses it...
        s = self.sds.source(self.src_path, "test")

    def test_test_method(self):
        # Verify that we can call a function.
        m = np.full((1, 2), 1)
        res = (
            self.sds.source(self.src_path, "test")
            .test_function(self.sds.full((1, 2), 1))[1]
            .as_matrix()
            .compute()
        )
        self.assertTrue(np.allclose(m, res))


if __name__ == "__main__":
    unittest.main(exit=False)
