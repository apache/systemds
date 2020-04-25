# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to you under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import onnx
from onnx import helper, numpy_helper


def save_graph(graph_def, name):
    model_def = helper.make_model(graph_def, producer_name="onnx-systemds test-graph generator")
    onnx.save_model(model_def, name)


def generate_simple_add_graph():
    A = helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable A")
    B = helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable B")
    C = helper.make_tensor_value_info('C', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable C")
    D = helper.make_tensor_value_info('D', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable D")
    E = helper.make_tensor_value_info('E', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable E")
    F = helper.make_tensor_value_info('F', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable F")

    nodes = [
        helper.make_node("Add", ['A', 'B'], ['C'], name="AddNodeName",
                         doc_string="This is a description of this Add operation"),
        helper.make_node("Add", ['C', 'D'], ['E'], name="MulNodeName",
                         doc_string="This is a description of this mul operation"),
        helper.make_node("Add", ['A', 'E'], ['F'], name="MulNodeName")
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="A simple matrix addition test graph",
        inputs=[A, B, D],
        outputs=[F],
        initializer=None,
        doc_string="Doc string of a simple matrix addition test graph",
        value_info=[C, E]
    )

    save_graph(graph, "simple_mat_add.onnx")


def generate_simple_mat_add_mul_sub_graph():
    A = helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable A")
    B = helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable B")
    C = helper.make_tensor_value_info('C', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable C")
    D = helper.make_tensor_value_info('D', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable D")
    E = helper.make_tensor_value_info('E', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable E")
    F = helper.make_tensor_value_info('F', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable F")

    nodes = [
        helper.make_node("Add", ['A', 'B'], ['C'], name="AddNodeName",
                         doc_string="This is a description of this Add operation"),
        helper.make_node("MatMul", ['C', 'D'], ['E'], name="MulNodeName",
                         doc_string="This is a description of this mul operation"),
        helper.make_node("Sub", ['A', 'E'], ['F'], name="MulNodeName")
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="A simple matrix addition, multiplication and substraction test graph",
        inputs=[A, B, D],
        outputs=[F],
        initializer=None,
        doc_string="Doc string with additional information",
        value_info=[C, E]
    )

    save_graph(graph, "simple_mat_add_mul_sub.onnx")


def generate_simple_initialized_graph():
    A_init = helper.make_tensor("A_init", onnx.TensorProto.FLOAT, [3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    B_init = helper.make_tensor("B_init", onnx.TensorProto.FLOAT, [3, 3], [2, 4, 6, 8, 10, 12, 14, 16, 18])

    B_init_valinfo = helper.make_tensor_value_info(B_init.name, onnx.TensorProto.FLOAT, B_init.dims,
                                                   doc_string="A single value tensor")
    A_init_valinfo = helper.make_tensor_value_info(A_init.name, onnx.TensorProto.FLOAT, A_init.dims,
                                                   doc_string="A 3x3 matrix")
    C = helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [3, 3], doc_string="This is the output C")
    D = helper.make_tensor_value_info("D", onnx.TensorProto.FLOAT, [3, 3], doc_string="This is the output D")

    nodes = [
        helper.make_node("Neg", ["A_init"], ["C"]),
        helper.make_node("Add", ["B_init", "C"], ["D"])
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="Simple mat initialized graph",
        inputs=[A_init_valinfo, B_init_valinfo],
        outputs=[D],
        initializer=[A_init, B_init],
        value_info=[A_init_valinfo, B_init_valinfo, C, D]
    )

    save_graph(graph, "simple_mat_initialized.onnx")


def generate_simple_boolean_noshape():
    A = helper.make_tensor_value_info('A', onnx.TensorProto.BOOL,
                                      doc_string="This is a description of variable A", shape=[])
    B = helper.make_tensor_value_info('B', onnx.TensorProto.BOOL,
                                      doc_string="This is a description of variable B", shape=[])
    C = helper.make_tensor_value_info('C', onnx.TensorProto.BOOL,
                                      doc_string="This is a description of variable C", shape=[])
    D = helper.make_tensor_value_info('D', onnx.TensorProto.BOOL,
                                      doc_string="This is a description of variable D", shape=[])
    E = helper.make_tensor_value_info('E', onnx.TensorProto.BOOL,
                                      doc_string="This is a description of variable E", shape=[])

    nodes = [
        helper.make_node("Or", ["A", "B"], ["C"]),
        helper.make_node("And", ["C", "A"], ["D"]),
        helper.make_node("Xor", ["B", "D"], ["E"])
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="Simple bool and or xor noshape graph",
        inputs=[A, B],
        outputs=[E],
        value_info=[A, B, C, D, E]
    )

    save_graph(graph, "simple_bool_and_or_xor_noshape.onnx")


def generate_simple_relu_tanh_sigmoid_softmax():
    A = helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable A")
    B = helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable B")
    C = helper.make_tensor_value_info('C', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable C")
    D = helper.make_tensor_value_info('D', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable D")
    E = helper.make_tensor_value_info('E', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable E")
    F = helper.make_tensor_value_info('F', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable F")
    G = helper.make_tensor_value_info('G', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable G")
    H = helper.make_tensor_value_info('H', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable H")

    nodes = [
        helper.make_node("Relu", ["A"], ["E"], doc_string="Call of Relu function"),
        helper.make_node("Tanh", ["B"], ["F"], doc_string="Call of Tanh function"),
        helper.make_node("Sigmoid", ["C"], ["G"], doc_string="Call of Sigmoid function"),
        helper.make_node("Softmax", ["D"], ["H"], doc_string="Call of Softmax function")
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="Simple relu tanh sigmoid softmax graph",
        inputs=[A, B, C, D],
        outputs=[E, F, G, H],
        value_info=[A, B, D, E, F, G, H],
        doc_string="This graph tests simple nn layer calls"
    )

    save_graph(graph, "simple_relu_tanh_sigmoid_softmax.onnx")


def generate_simple_dropout_layer():
    x = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable x")
    y = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable y")
    z = helper.make_tensor_value_info('z', onnx.TensorProto.FLOAT, [2, 2],
                                      doc_string="This is a description of variable z")
    mask = helper.make_tensor_value_info('mask', onnx.TensorProto.BOOL, [2, 2],
                                         doc_string="This is a description of variable mask")

    nodes = [
        onnx.helper.make_node(
            op_type='Dropout',
            inputs=['x'],
            outputs=['y'],
            ratio=.1
        ),
        onnx.helper.make_node(
            op_type='Dropout',
            inputs=['y'],
            outputs=['z', 'mask'],
            ratio=.1
        )
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="Simple dropout graph",
        inputs=[x],
        outputs=[z, mask],
        value_info=[x, y, z],
        doc_string="This graph tests a simple dropout layer call"
    )

    save_graph(graph, "simple_dropout_layer.onnx")


if __name__ == '__main__':
    generate_simple_add_graph()
    generate_simple_mat_add_mul_sub_graph()
    generate_simple_initialized_graph()
    generate_simple_relu_tanh_sigmoid_softmax()
    generate_simple_dropout_layer()

    # generate_simple_boolean_noshape()
