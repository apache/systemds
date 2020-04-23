import ntpath
import os
import re

import onnx
import onnx_model
from onnx_model import OnnxModel
import jinja2
import value_prepare
import operator_gen


# functions in this file contain the logic for rendering the dml script

def gen_script_from_node(env: jinja2.environment.Environment, node: onnx.NodeProto) -> str:
    # TODO handle unsupported operators
    script_to_operator = {
        "Add": operator_gen.gen_simple_2input_1output_mat_operator,
        "Sub": operator_gen.gen_simple_2input_1output_mat_operator,
        "MatMul": operator_gen.gen_simple_2input_1output_mat_operator,
        "Neg": operator_gen.gen_simple_1input_1output_mat_operator
    }
    try:
        return script_to_operator[node.op_type](env, node)
    except KeyError as error:
        print("Operator " + str(node.op_type) + " not supported")
        raise error


def gen_node_script_graph(env: jinja2.environment.Environment, model: OnnxModel) -> [onnx_model.TreeNode]:
    """
    Traverses the node tree of the OnnxModel structure and generates a script string for each node,
    the returned node list is ordered such that it can be inserted into the finished script
    :param env: Jinja environment to load the template files
    :param model: the internal model structure
    :return: list of nodes with inserted node scripts in sorted order
    """
    # 1. get lowest nodes
    # 2. check if all outputs of nodes are computed
    # 3. insert into graph
    # TODO: handle if conditional and for loop
    generated_nodes = []
    current_nodes = model.node_tree
    available_outputs = [o.name for o in list(model.get_graph_outputs())]
    while len(current_nodes.nodes) != 0:
        current_lowest_nodes = current_nodes.end_nodes

        # Find next operation to insert
        next_node: onnx_model.TreeNode = None
        for node in current_lowest_nodes:
            if all(output in available_outputs for output in list(node.node.output)):
                next_node = node
                break
        if not next_node:
            raise Exception("Error in parsing nodes, did not find a next node to compute")
        next_node.generated_script = gen_script_from_node(env, next_node.node)

        # After insertion the inputs to the node become available outputs and we remove the node
        available_outputs += list(next_node.node.input)
        current_nodes.remove_end_node(next_node)
        generated_nodes.append(next_node)

    generated_nodes.reverse()
    return generated_nodes


def gen_model_header(env: jinja2.environment.Environment, model: OnnxModel) -> str:
    header_template = env.get_template("model_header.dml.jinja")
    onnx_model = model.onnx_model
    header_infos = dict()

    header_infos["ir_version"] = onnx_model.ir_version
    opset_import = list()
    for opset in onnx_model.opset_import:
        if len(opset.domain) == 0:
            opset.domain = "ONNX"
        opset_import.append(opset.domain + "/" + str(opset.version))
    header_infos["producer_name"] = onnx_model.producer_name
    header_infos["producer_version"] = onnx_model.producer_version
    header_infos["domain"] = onnx_model.domain
    header_infos["model_version"] = onnx_model.model_version
    header_infos["doc_string"] = onnx_model.doc_string
    metadata_props = list()
    for prop in onnx_model.metadata_props:
        metadata_props.append([prop.key, prop.value])

    model_header_render = header_template.render(
        test=onnx_model,
        header_components=header_infos,
        opset_import=opset_import,
        metadata_props=metadata_props
    )
    return model_header_render


def gen_graph_function(env: jinja2.environment.Environment, model: OnnxModel,
                       generated_node_scripts: [onnx_model.TreeNode]) -> str:
    onnx_graph = model.onnx_model.graph
    function_template = env.get_template("graph_function.dml.jinja")

    inputs_with_initializers = model.get_graph_inputs_with_initializers()
    inputs_without_initializers = model.get_graph_inputs_without_initializers()
    outputs = model.get_graph_outputs()

    # TODO: do more compatibility checks
    # parse tensor
    function_inputs = value_prepare.prepare_function_inputs(inputs_without_initializers)
    function_outputs = value_prepare.prepare_function_outputs(outputs)
    function_initializers = value_prepare.prepare_initialized_inputs(inputs_with_initializers)

    # generate function name from graph name
    function_name = "gen_" + re.sub(r"[-| ]", "_", onnx_graph.name.lower())
    function_name = re.sub(r"[^0-9a-z_]", "", function_name)

    graph_function_render = function_template.render(
        function_inputs=function_inputs,
        function_outputs=function_outputs,
        function_start_initializers=function_initializers,
        graph_function_name=function_name,
        graph_function_description=onnx_graph.doc_string,
        node_scripts=[node.generated_script for node in generated_node_scripts]
    )
    return graph_function_render


def write_script(model: OnnxModel, output_file: str) -> str:
    """
    Generate the dml script from the internal model structure
    :param model: The model from which the dml script shall be created
    :param output_file: The file to which the script shall be written
    :return: The dml script which is written to the output file
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(current_dir + '/templates/'))

    model_header_render = gen_model_header(env, model)
    operator_render_list = gen_node_script_graph(env, model)
    graph_function = gen_graph_function(env, model, operator_render_list)

    main_template = env.get_template("main.dml.jinja")
    result_render = main_template.render(
        title="This file was generated by onnx-systemds from the input-file " + ntpath.basename(model.input_file),
        model_header_render=model_header_render,
        graph_render=graph_function
    )
    with open(output_file, 'w') as f:
        f.write(result_render)
    return result_render
