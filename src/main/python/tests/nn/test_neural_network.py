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
from tests.nn.neural_network import NeuralNetwork
from systemds.script_building.script import DMLScript


class TestNeuralNetwork(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()
        np.random.seed(42)
        cls.X = np.random.rand(6, 1)
        cls.exp_out = np.array(
            [
                -0.37768756,
                -0.47785831,
                -0.95870362,
                -1.21297214,
                -0.73814523,
                -0.933917,
                -0.60368929,
                -0.76380049,
                -0.15732974,
                -0.19905692,
                -0.15730542,
                -0.19902615,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_forward_pass(self):
        Xm = self.sds.from_numpy(self.X)
        nn = NeuralNetwork(self.sds, dim=1)
        # test forward pass through the network using static calls
        static_out = nn.forward_static_pass(Xm).compute().flatten()

        self.assertTrue(np.allclose(static_out, self.exp_out))

        # test forward pass through the network using dynamic calls
        dynamic_out = nn.forward_dynamic_pass(Xm).compute().flatten()
        self.assertTrue(np.allclose(dynamic_out, self.exp_out))

    def test_multiple_sourcing(self):
        sds = SystemDSContext()
        Xm = sds.from_numpy(self.X)
        nn = NeuralNetwork(sds, dim=1)

        # test for verifying that affine and relu are each being sourced exactly once
        network_out = nn.forward_static_pass(Xm)
        scripts = DMLScript(sds)
        scripts.build_code(network_out)

        self.assertEqual(
            1, self.count_sourcing(scripts.dml_script, layer_name="affine")
        )
        self.assertEqual(1, self.count_sourcing(scripts.dml_script, layer_name="relu"))
        sds.close()

    def count_sourcing(self, script: str, layer_name: str):
        """
        Count the number of times the dml script is being sourced
        i.e. count the number of occurrences of lines like
        'source(...) as relu' in the dml script

        :param script: the sourced dml script text
        :param layer_name: example: "affine", "relu"
        :return:
        """
        return len(
            [
                line
                for line in script.split("\n")
                if all([line.startswith("source"), line.endswith(layer_name)])
            ]
        )


if __name__ == "__main__":
    unittest.main()
