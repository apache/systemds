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

import os
import re

import util
import onnx
import onnx_helper
import jinja2
import operator_gen


def gen_script_from_node(env: jinja2.environment.Environment, node: onnx.NodeProto) -> (str, str):
    """
    Generates a dml script snippet and the required imports for the given node

    :param env: Jinja environment to load the template files
    :param node: the node for which the dml snippet shall be generated
    :return: Tuple of (imports, script)
    """
    script_to_operator = {
        "Add": operator_gen.gen_simple_2input_1output_operator,
        "Sub": operator_gen.gen_simple_2input_1output_operator,
        "MatMul": operator_gen.gen_simple_2input_1output_operator,
        "Neg": operator_gen.gen_simple_1input_1output_mat_operator,
        "Xor": operator_gen.gen_function_call,
        "Or": operator_gen.gen_simple_2input_1output_operator,
        "And": operator_gen.gen_simple_2input_1output_operator,
        "Relu": operator_gen.gen_function_call,
        "Tanh": operator_gen.gen_function_call,
        "Sigmoid": operator_gen.gen_function_call,
        "Softmax": operator_gen.gen_function_call,
        "Dropout": operator_gen.gen_dropout_call,
    }
    try:
        return script_to_operator[node.op_type](env, node)
    except KeyError as error:
        print("Operator " + str(node.op_type) + " not supported")
        raise error


def gen_node_scripts(env: jinja2.environment.Environment, graph: onnx.GraphProto) -> ([str], [str]):
    """
    Traverses the node tree of the onnx-graph structure and generates a script string for each node,
    as well as a string for the required imports.
    The lists are correctly ordered for inserting them in the dml script.

    :param env: Jinja environment to load the template files
    :param graph: the onnx graph for which the script shall be generated
    :return: two lists (imports, scripts)
    """
    # 1. get lowest nodes
    # 2. check if all outputs of nodes are computed
    # 3. insert into graph
    # TODO: handle if conditional and for loop
    generated_scripts = []
    generated_imports = set()  # set to avoid duplicate imports
    node_tree = onnx_helper.NodeTree(graph.node)
    available_outputs = [o.name for o in list(graph.output)]
    while len(node_tree.nodes) != 0:
        current_lowest_nodes = node_tree.end_nodes

        # Find next operation to insert
        next_tree_node = None
        for tree_node in current_lowest_nodes:
            if all(output in available_outputs for output in list(tree_node.node.output)):
                next_tree_node = tree_node
                break
        if not next_tree_node:
            raise Exception("Error in parsing nodes, did not find a next node to compute")
        generated_import, generated_script = gen_script_from_node(env, next_tree_node.node)

        # After insertion the inputs to the node become available outputs and we remove the node
        available_outputs += list(next_tree_node.node.input)
        node_tree.remove_end_node(next_tree_node)
        generated_scripts.append(generated_script)
        if len(generated_import) > 0:
            generated_imports.add(generated_import)

    generated_scripts.reverse()
    return list(generated_imports), generated_scripts


def gen_model_header(env: jinja2.environment.Environment, onnx_model: onnx.ModelProto) -> str:
    """
    Generates the header of the script for the given model

    :param env: Jinja environment to load the template files
    :param onnx_model: the onnx model for which the header shall be generated
    :return: the generated header
    """
    header_template = env.get_template("model_header.dml.jinja")
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


def gen_graph_function(env: jinja2.environment.Environment, graph: onnx.GraphProto,
                       generated_node_scripts: [str]) -> str:
    """
    Generates the dml function for the given onnx-graph and inserts the node scripts in
    the function-body.

    :param env: Jinja environment to load the template files
    :param graph: the onnx-graph for which the function shall be generated
    :param generated_node_scripts: the node scripts in correct order for the function-body
    :return: the generated dml-function
    """
    function_template = env.get_template("graph_function.dml.jinja")

    inputs_with_initializers = onnx_helper.get_graph_inputs_with_initializers(graph)
    inputs_without_initializers = onnx_helper.get_graph_inputs_without_initializers(graph)
    outputs = list(graph.output)

    # parse values
    function_inputs = onnx_helper.prepare_function_inputs(inputs_without_initializers)
    function_outputs = onnx_helper.prepare_function_outputs(outputs)
    function_initializers = onnx_helper.prepare_initialized_inputs(inputs_with_initializers)

    # generate function name from graph name
    function_name = "gen_" + re.sub(r"[-| ]", "_", graph.name.lower())
    function_name = re.sub(r"[^0-9a-z_]", "", function_name)

    # render main file
    graph_function_render = function_template.render(
        function_inputs=function_inputs,
        function_outputs=function_outputs,
        function_start_initializers=function_initializers,
        graph_function_name=function_name,
        graph_function_description=graph.doc_string,
        node_scripts=generated_node_scripts
    )
    return graph_function_render


def gen_script(onnx_model: onnx.ModelProto, output_file: str = None) -> str:
    """
    Generate the dml script for the given model and return the generated string.
    If an output_file is given, the script is also written to a file.

    :param onnx_model: the model for which the dml script shall be generated
    :param output_file: (optional) the file to which the script shall be written
    :return: the generated dml-script
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(current_dir + '/templates/'))

    model_header_render = gen_model_header(env, onnx_model)
    node_imports, node_scripts = gen_node_scripts(env, onnx_model.graph)
    graph_function = gen_graph_function(env, onnx_model.graph, node_scripts)

    wdir = ""
    if len(node_imports) > 0:
        # need to set wdir to enable imports
        wdir = util.resolve_systemds_root() + "scripts"

    main_template = env.get_template("main.dml.jinja")
    result_render = main_template.render(
        title="This file was generated by onnx-systemds",
        model_header_render=model_header_render,
        wdir=wdir,
        imports=node_imports,
        graph_render=graph_function
    )
    if output_file:
        directory = os.path.dirname(output_file)
        if len(directory) > 0:
            os.makedirs(directory, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(result_render)

    return result_render
