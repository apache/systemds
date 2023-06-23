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

from systemds.operator.nn_nodes.relu import ReLU

X = np.array([0, -1, -2, 2, 3, -5])
dout = np.array([0, 1, 2, 3, 4, 5])


class TestRelu(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_forward(self):
        relu = ReLU()
        #forward
        Xm = self.sds.from_numpy(X)
        out = relu.forward(Xm).compute().flatten()
        expected = np.array([0, 0, 0, 2, 3, 0])
        self.assertTrue(np.allclose(out, expected))

    def test_backward(self):
        relu = ReLU()
        # forward
        Xm = self.sds.from_numpy(X)
        out = relu.forward(Xm)
        # backward
        doutm = self.sds.from_numpy(dout)
        dx = relu.backward(doutm).compute().flatten()
        expected = np.array([0, 0, 0, 3, 4, 0], dtype=np.double)
        self.assertTrue(np.allclose(dx, expected))

if __name__ == '__main__':
    unittest.main()
