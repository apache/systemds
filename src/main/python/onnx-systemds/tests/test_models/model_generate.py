import numpy as np
import onnx
from onnx import helper, numpy_helper


def save_graph(graph_def, name):
    model_def = helper.make_model(graph_def, producer_name="onnx-systemds test-graph generator")
    model_def.opset_import[0].version = 7
    model_def.ir_version = 3
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


if __name__ == '__main__':
    generate_simple_add_graph()
    generate_simple_mat_add_mul_sub_graph()
    generate_simple_initialized_graph()
