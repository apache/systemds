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

import jinja2
import onnx


def gen_function_call(env: jinja2.environment.Environment, node: onnx.NodeProto) -> (str, str):
    operator_template = env.get_template("operators/" + "function_call.dml.jinja")
    import_template = env.get_template("module_import.dml.jinja")

    if len(list(node.output)) != 1:
        raise Exception("Function call needs output")

    if len(node.attribute) != 0:
        raise Exception("attributes not supported for operator")

    required_import = {
        "Relu": {"path": "/nn/layers/relu.dml", "import_name": "relu_layer", "function_name": "forward"},
        "Tanh": {"path": "/nn/layers/tanh.dml", "import_name": "tanh_layer", "function_name": "forward"},
        "Sigmoid": {"path": "/nn/layers/sigmoid.dml", "import_name": "sigmoid_layer", "function_name": "forward"},
        "Softmax": {"path": "/nn/layers/softmax.dml", "import_name": "softmax_layer", "function_name": "forward"}
    }

    import_render = ""
    function_name = node.op_type
    function_namespace = ""
    if node.op_type in required_import.keys():
        module_import = required_import[node.op_type]
        import_render = import_template.render(
            path=module_import["path"],
            name=module_import["import_name"]
        )
        function_name = module_import["function_name"]
        function_namespace = module_import["import_name"]

    node_render = operator_template.render(
        function_namespace=function_namespace,
        function=function_name,
        arguments=list(node.input),
        outputs=list(node.output),
        doc_string=node.doc_string
    )
    return import_render, node_render


def gen_simple_2input_1output_operator(env: jinja2.environment.Environment, node: onnx.NodeProto) -> (str, str):
    operator = {
        "Add": "+",
        "Sub": "-",
        "MatMul": "%*%",
        "And": "&",
        "Or": "|"
    }
    operator_template = env.get_template("operators/2input_1output_operator.dml.jinja")

    if len(node.attribute) != 0:
        raise Exception("attributes not supported for operator")

    if len(list(node.input)) > 2 or len(list(node.output)) > 1:
        raise Exception("Operator needs 2 inputs and 1 output")

    node_render = operator_template.render(
        input_0=list(node.input)[0],
        input_1=list(node.input)[1],
        output=list(node.output)[0],
        operator=operator[node.op_type],
        doc_string=node.doc_string
    )
    return "", node_render


def gen_simple_1input_1output_mat_operator(env: jinja2.environment.Environment, node: onnx.NodeProto) -> (str, str):

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
    return "", node_render


def gen_dropout_call(env: jinja2.environment.Environment, node: onnx.NodeProto):
    operator_template = env.get_template("operators/" + "function_call.dml.jinja")
    import_template = env.get_template("module_import.dml.jinja")

    function_namespace = "dropout_layer"
    function_name = "forward"
    path = "/nn/layers/dropout.dml"

    #  * Inputs:
    #    *  - X: Inputs, of shape (any, any).
    #    *  - p: Probability of keeping a neuron output.
    #    *  - seed: [Optional: -1] Random number generator seed to allow for
    #    *      deterministic evaluation.  Set to -1 for a random seed.
    # * Outputs:
    #    *  - out: Outputs, of same shape as `X`.
    #    *  - mask: Dropout mask used to compute the output.

    X = list(node.input)[0]
    p = 0.5
    seed = -1
    if len(list(node.attribute)) > 0:
        attributes = list(node.attribute)
        if attributes[0].name != "ratio" or len(attributes) > 1:
            raise Exception("Error in generating dropout call invalid attributes" + str(attributes))
        p = attributes[0].f

    import_render = import_template.render(
        path=path,
        name=function_namespace
    )

    node_render = operator_template.render(
        function_namespace=function_namespace,
        function=function_name,
        arguments=[X, p, seed],
        outputs=list(node.output),
        doc_string=node.doc_string
    )
    return import_render, node_render
