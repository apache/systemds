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
import tests.onnx_systemds.util as util


class TestSimpleOperators(unittest.TestCase):
    def test_simple_mat_add(self):
        name = "simple_mat_add"
        util.run_and_compare_output(name, self)

    def test_simple_mat_add_mul_sub(self):
        name = "simple_mat_add_mul_sub"
        util.run_and_compare_output(name, self)

    def test_simple_mat_initialized(self):
        name = "simple_mat_initialized"
        util.run_and_compare_output(name, self)

    def test_simple_relu_tanh_sigmoid_softmax(self):
        name = "simple_relu_tanh_sigmoid_softmax"
        util.run_and_compare_output(name, self)

    def test_simple_conv2d_layer(self):
        name = "simple_conv_layer"
        util.run_and_compare_output(name, self)

    def test_simple_conv2d_layer_2(self):
        name = "simple_conv_layer_2"
        util.run_and_compare_output(name, self)

    def test_simple_maxpool_layer(self):
        name = "simple_maxpool_layer"
        util.run_and_compare_output(name, self)

    def test_simple_if_graph(self):
        name = "simple_if_graph"
        util.run_and_compare_output(name, self)

    # TODO: dml implementation of dropout does not work
    # def test_simple_dropout_layer(self):
    #     name = "simple_dropout_layer"
    #     test_util.run_and_compare_output(name, self)

    # TODO: dml does not support boolean matrices?
    # def test_simple_bool_and_or_xor_noshape(self):
    #     name = "simple_bool_and_or_xor_noshape"
    #     test_util.run_and_compare_output(name, self)


if __name__ == '__main__':
    unittest.main()
