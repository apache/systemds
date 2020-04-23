import onnx_model, onnx
import jinja2


# functions in this file contain operator specific logic for rendering the dml script

def gen_simple_2input_1output_mat_operator(env: jinja2.environment.Environment, node: onnx.NodeProto) -> str:

    template_for_operator = {
        "Add": "add.dml.jinja",
        "Sub": "sub.dml.jinja",
        "MatMul": "matmul.dml.jinja"
    }

    operator_template = env.get_template("operators/" + template_for_operator[node.op_type])

    if len(node.attribute) != 0:
        raise Exception("attributes not supported for operator")

    if len(list(node.input)) > 2 or len(list(node.output)) > 1:
        raise Exception("Operator needs 2 inputs and 1 output")

    node_render = operator_template.render(
        input_0=list(node.input)[0],
        input_1=list(node.input)[1],
        output=list(node.output)[0],
        doc_string=node.doc_string
    )
    return node_render


def gen_simple_1input_1output_mat_operator(env: jinja2.environment.Environment, node: onnx.NodeProto) -> str:

    template_for_operator = {
        "Neg": "neg.dml.jinja",
    }

    operator_template = env.get_template("operators/" + template_for_operator[node.op_type])

    if len(node.attribute) != 0:
        raise Exception("attributes not supported for operator")

    if len(list(node.input)) != 1 or len(list(node.output)) != 1:
        raise Exception("Operator needs 1 input and 1 output")

    node_render = operator_template.render(
        input=list(node.input)[0],
        output=list(node.output)[0],
        doc_string=node.doc_string
    )
    return node_render
