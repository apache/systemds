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
from functools import reduce

import onnx
import onnx.version_converter


class TreeNode:
    def __init__(self, node: onnx.NodeProto):
        self.node = node
        self.parent_nodes = list()
        self.child_nodes = list()


class NodeTree:
    """ A simple class for representing a tree structure of nodes """
    def __init__(self, nodes: [onnx.NodeProto]):
        self.nodes = [TreeNode(node) for node in nodes]
        self.root_nodes = []  # nodes that have no parents
        self.end_nodes = []   # nodes that have no children

        # find parents and children for each node
        for tree_node in self.nodes:
            for compare_tree_node in self.nodes:
                if tree_node != compare_tree_node:
                    for node_output in tree_node.node.output:
                        if node_output in compare_tree_node.node.input:
                            tree_node.child_nodes.append(compare_tree_node)
                            compare_tree_node.parent_nodes.append(tree_node)

        for node in self.nodes:
            if len(node.child_nodes) == 0:
                self.end_nodes.append(node)
            if len(node.parent_nodes) == 0:
                self.root_nodes.append(node)

    def remove_end_node(self, node: TreeNode):
        if node not in self.end_nodes:
            raise Exception("Can only remove end nodes")
        self.end_nodes.remove(node)
        self.nodes.remove(node)

        for parent_node in node.parent_nodes:
            parent_node.child_nodes.remove(node)
        self.end_nodes += node.parent_nodes
        node.parent_nodes = []


def load_model(onnx_file: str) -> onnx.ModelProto:
    """
    Loads the onnx file, checks the model and converts it to a common version.

    :param onnx_file:
    :return: the loaded onnx-model
    """
    TARGET_VERSION = 11
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)
    if len(list(model.opset_import)) == 1 and list(model.opset_import)[0].version == TARGET_VERSION:
        return model
    else:
        return onnx.version_converter.convert_version(model, TARGET_VERSION)


def get_value_info(graph: onnx.GraphProto, name: str) -> onnx.ValueInfoProto:
    """
    Searches the `graph` for the given `name` and returns the associated ValueInfo,
    if the name is not found None is returned

    :param graph: the onnx-graph that shall be searched
    :param name: the name for of the value
    :return: the value-info or None if it is not found
    """
    for info in graph.input:
        if info.name == name:
            return info

    for info in graph.value_info:
        if info.name == name:
            return info

    for info in graph.output:
        if info.name == name:
            return info

    return None


def get_valueinfo_dimensions(value_info: onnx.ValueInfoProto) -> [int]:
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def get_graph_inputs_without_initializers(graph: onnx.GraphProto) -> [onnx.ValueInfoProto]:
    """
    Returns all inputs of the `graph` that have no associated initializer values.

    :param graph: the onnx-graph
    :return: list of uninitialized inputs
    """
    inputs_without_initializers = []
    for input in graph.input:
        has_initializer = False
        for initializer in graph.initializer:
            if initializer.name == input.name:
                has_initializer = True
                break

        if not has_initializer:
            inputs_without_initializers.append(input)

    return inputs_without_initializers


def get_graph_inputs_with_initializers(graph: onnx.GraphProto) -> [(onnx.ValueInfoProto, onnx.TensorProto)]:
    """
    Returns all initialized inputs of the `graph` with their corresponding initializer.

    :param graph: the onnx-graph
    :return: list of tuples of (input, initializer)
    """
    inputs_with_initializers = []

    for input in graph.input:
        for initializer in graph.initializer:
            if initializer.name == input.name:
                inputs_with_initializers.append((input, initializer))

    return inputs_with_initializers


class PreparedValue:
    """ Class for preparing onnx value structures for writing them to the dml script """
    def __init__(self, value_info: onnx.ValueInfoProto, initializer: onnx.TensorProto = None):

        supported_types = ["int", "boolean", "double"]

        # TODO: these type translations are not correct double -> float
        type_translation = {
            1: "double",  # float
            2: "int",  # uint8_t
            3: "int",  # int8_t
            4: "int",  # uint16_t
            5: "int",  # int16_t
            6: "int",  # int32_t
            7: "int",  # int64_t
            8: "string",
            9: "boolean", # bool

            10: "double",  # float16,
            11: "double",
            12: "int",  # uint32
            13: "int",  # uint64

            14: "COMPLEX64",
            15: "COMPLEX128",
            16: "BFLOAT16"
        }

        if value_info.type.tensor_type.elem_type not in type_translation.keys():
            raise NotImplementedError("Only support Tensor Types")

        # TODO: add support for other data types

        self.data_type = "matrix"
        self.value_type = type_translation[value_info.type.tensor_type.elem_type]
        if self.value_type not in supported_types:
            raise NotImplementedError("The type " + self.value_type + " is currently not supported")

        self.shape = []
        shape_dimensions = value_info.type.tensor_type.shape.dim

        for dim in shape_dimensions:
            # TODO: shapes with no value but instead name -> support?
            if len(dim.dim_param) != 0:
                raise NotImplementedError("Only support dim_value")
            self.shape.append(dim.dim_value)

        if len(self.shape) > 2:
            # TODO: not sure this is the solution for every instance of this problem
            # Multiply all shapes right
            rows = self.shape[0]
            cols = reduce(lambda s0, s1: s0*s1, self.shape[1:])
            self.shape = [rows, cols]

        self.identifier_name = value_info.name
        self.description = value_info.doc_string
        self.initializer = None

        if initializer:
            self.initializer_values = list(initializer.float_data)


def prepare_function_inputs(inputs: [onnx.ValueInfoProto]) -> [PreparedValue]:
    return [PreparedValue(i) for i in inputs]


def prepare_initialized_inputs(inputs: [(onnx.ValueInfoProto, onnx.TensorProto)]) -> [PreparedValue]:
    return [PreparedValue(info, init) for info, init in inputs]


def prepare_function_outputs(outputs: [onnx.ValueInfoProto]) -> [PreparedValue]:
    return [PreparedValue(o) for o in outputs]



