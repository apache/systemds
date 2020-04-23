import onnx
import onnx.version_converter

# This file contains a data structure that stores the base onnxmodel
# together with helper functions for parsing the onnx data structure


class TreeNode:
    def __init__(self, node):
        self.node = node
        self.parent_nodes = list()
        self.child_nodes = list()
        self.generated_script = None

    def get_input_names(self) -> [str]:
        return self.node.input


class NodeTree:
    def __init__(self, nodes: [onnx.NodeProto]):
        self.nodes = [TreeNode(node) for node in nodes]
        self.root_nodes = []
        self.end_nodes = []

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
        return node


class OnnxModel:
    def __init__(self, onnx_file):
        self.MODEL_VERSION = 7
        self.input_file = onnx_file
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)
        converted_model = onnx.version_converter.convert_version(model, self.MODEL_VERSION)
        self.onnx_model = converted_model
        self.onnx_graph = converted_model.graph

        # Build tree structure
        self.node_tree = NodeTree(list(converted_model.graph.node))

    def get_graph_inputs_without_initializers(self) -> [onnx.ValueInfoProto]:
        graph = self.onnx_model.graph
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

    def get_graph_inputs_with_initializers(self) -> [(onnx.ValueInfoProto, onnx.TensorProto)]:
        graph = self.onnx_model.graph
        inputs_with_initializers = []

        for input in graph.input:
            for initializer in graph.initializer:
                if initializer.name == input.name:
                    inputs_with_initializers.append((input, initializer))

        return inputs_with_initializers

    def get_graph_outputs(self) -> [onnx.ValueInfoProto]:
        return self.onnx_model.graph.output

