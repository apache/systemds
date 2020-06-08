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

import os

import jinja2
import onnx
import systemds.onnx_systemds.onnx_helper as onnx_helper
from systemds.onnx_systemds import operator_gen, util

# Each operator listed shall be supported by this converter
operator_generators = {
    "Add": operator_gen.gen_2input_1output_operator,
    "Sub": operator_gen.gen_2input_1output_operator,
    "MatMul": operator_gen.gen_2input_1output_operator,
    "Neg": operator_gen.gen_1input_1output_mat_operator,
    "Xor": operator_gen.gen_simple_function_call,
    "Or": operator_gen.gen_2input_1output_operator,
    "And": operator_gen.gen_2input_1output_operator,
    "Relu": operator_gen.gen_simple_function_call,
    "Tanh": operator_gen.gen_simple_function_call,
    "Sigmoid": operator_gen.gen_simple_function_call,
    "Softmax": operator_gen.gen_simple_function_call,
    "Dropout": operator_gen.gen_dropout_call,
    "MaxPool": operator_gen.gen_maxpool_call,
    "Conv": operator_gen.gen_conv_call,
    "If": operator_gen.gen_if_call,
}


def gen_node_script(env: jinja2.environment.Environment, graph: onnx.GraphProto, node: onnx.NodeProto) \
        -> operator_gen.GeneratedScriptPart:
    """
    Generates a dml script snippet, the required imports and sub-graphs for the given `node`

    :param env: Jinja environment to load the template files
    :param graph: the onnx graph for which the script shall be generated
    :param node: the node for which the dml snippet shall be generated
    :return: The generated script-part
    """
    try:
        return operator_generators[node.op_type](env, graph, node)
    except KeyError as error:
        print("Operator " + str(node.op_type) + " not supported")
        raise error


def gen_graph_functions(env: jinja2.environment.Environment, main_graph: onnx.GraphProto) -> ([str], str, [str]):
    """
    Traverses the node tree of the onnx-graph structure and generates a script string for each node,
    as well as a string for the required imports together with all functions of sub-graphs.
    The resulting lists are correctly ordered for inserting them in the dml script.

    :param env: Jinja environment to load the template files
    :param main_graph: the onnx graph for which the script shall be generated
    :return: Tuple (imports, main function, sub-graph functions)
    """

    main_function_node_scripts = []
    sub_graph_functions = []
    generated_imports = set()  # set to avoid duplicate imports

    node_tree = onnx_helper.NodeTree(main_graph.node)
    available_outputs = [o.name for o in list(main_graph.output)]

    while len(node_tree.nodes) != 0:
        current_lowest_nodes = node_tree.end_nodes

        # Find next operation to insert -> check if all outputs are available
        next_tree_node = None
        for tree_node in current_lowest_nodes:
            if all(output in available_outputs for output in list(tree_node.node.output)):
                next_tree_node = tree_node
                break
        if not next_tree_node:
            raise Exception("Error in parsing nodes, did not find a next node to compute")

        # Insert generated parts
        generated_node = gen_node_script(env, main_graph, next_tree_node.node)
        generated_imports.update(generated_node.imports)
        main_function_node_scripts.append(generated_node.dml_script)
        # handle sub-graphs
        for sub_graph in generated_node.sub_graphs:
            sub_graph_imports, sub_graph_main_function, sub_graph_sub_graph_functions = \
                gen_graph_functions(env, sub_graph)
            # Inherit imports
            generated_imports.update(sub_graph_imports)
            # Inherit sub-graph functions of sub-graph
            sub_graph_functions += sub_graph_sub_graph_functions
            # Sub-graph main-function becomes sub-graph function
            sub_graph_functions.append(sub_graph_main_function)

        # After insertion the inputs to the node become available and the node is removed
        available_outputs += list(next_tree_node.node.input)
        node_tree.remove_end_node(next_tree_node)

    main_function_node_scripts.reverse()
    main_graph_function = render_function(env, main_graph, main_function_node_scripts)
    return list(generated_imports), main_graph_function, sub_graph_functions


def render_function(env: jinja2.environment.Environment, graph: onnx.GraphProto,
                    generated_node_scripts: [str]) -> str:
    """
    Generates the dml function for the given `graph` and inserts the 'generated_node_scripts' in
    the function-body.

    :param env: Jinja environment to load the template files
    :param graph: the graph for which the function shall be generated
    :param generated_node_scripts: the node scripts in correct order for the function-body
    :return: the generated function
    """
    function_template = env.get_template("graph_function.dml.jinja")

    inputs_with_initializers = onnx_helper.get_graph_inputs_with_initializers(graph)
    inputs_without_initializers = onnx_helper.get_graph_inputs_without_initializers(graph)
    outputs = list(graph.output)

    # prepare inputs/outputs
    function_inputs = [onnx_helper.PreparedValue(i) for i in inputs_without_initializers]
    function_outputs = [onnx_helper.PreparedValue(o) for o in outputs]
    function_initializers = [onnx_helper.PreparedValue(info, init) for info, init in inputs_with_initializers]

    # render function
    graph_function_render = function_template.render(
        function_inputs=function_inputs,
        function_outputs=function_outputs,
        function_start_initializers=function_initializers,
        graph_function_name=util.generate_function_name(graph.name),
        graph_function_description=graph.doc_string,
        node_scripts=generated_node_scripts
    )
    return graph_function_render


def gen_model_header(env: jinja2.environment.Environment, model: onnx.ModelProto) -> str:
    """
    Generates the header of the script for the given `model`

    :param env: Jinja environment to load the template files
    :param model: the onnx model for which the header shall be generated
    :return: the generated header
    """
    header_template = env.get_template("model_header.dml.jinja")
    header_infos = dict()

    header_infos["ir_version"] = model.ir_version
    opset_import = list()
    for opset in model.opset_import:
        if len(opset.domain) == 0:
            opset.domain = "ONNX"
        opset_import.append(opset.domain + "/" + str(opset.version))
    header_infos["producer_name"] = model.producer_name
    header_infos["producer_version"] = model.producer_version
    header_infos["domain"] = model.domain
    header_infos["model_version"] = model.model_version
    header_infos["doc_string"] = model.doc_string
    metadata_props = [[prop.key, prop.vale] for prop in model.metadata_props]

    model_header_render = header_template.render(
        header_components=header_infos,
        opset_import=opset_import,
        metadata_props=metadata_props
    )
    return model_header_render


def gen_script(model: onnx.ModelProto, output_file: str = None) -> str:
    """
    Generate the dml script for the given `model` and return it.
    If an `output_file` is given, the script is also written to a file.

    :param model: the model for which the dml script shall be generated
    :param output_file: (optional) the file to which the script shall be written
    :return: the generated dml-script
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(current_dir + '/templates/'))
    model_header_render = gen_model_header(env, model)
    imports, main_function, sub_functions = gen_graph_functions(env, model.graph)

    wdir = ""
    if len(imports) > 0:
        # need to set wdir to enable imports
        wdir = util.resolve_systemds_root() + "/scripts"

    main_template = env.get_template("main.dml.jinja")
    result_render = main_template.render(
        title="This file was generated by onnx-systemds",
        model_header_render=model_header_render,
        wdir=wdir,
        imports=imports,
        main_function=main_function,
        sub_functions=sub_functions
    )
    if output_file:
        directory = os.path.dirname(output_file)
        if len(directory) > 0:
            os.makedirs(directory, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(result_render)

    return result_render
