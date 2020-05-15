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

from random import randint

import jinja2
import onnx
import systemds.onnx_systemds.onnx_helper as onnx_helper
from systemds.onnx_systemds import util


class GeneratedScriptPart:
    def __init__(self, dml_script: str, imports: [str] = None, sub_graphs: [onnx.GraphProto] = None):
        if sub_graphs is None:
            sub_graphs = []
        if imports is None:
            imports = []
        self.dml_script = dml_script
        self.imports = imports
        self.sub_graphs = sub_graphs


def gen_simple_function_call(env: jinja2.environment.Environment, graph: onnx.GraphProto,
                             node: onnx.NodeProto) -> GeneratedScriptPart:
    """
    Generates a simple function call by directly providing the node inputs as arguments
    and node outputs as outputs to a function call. Additionally adds the required imports.

    :param env: Jinja environment to load the template files
    :param graph: the onnx-graph for which the script shall be generated
    :param node: the onnx-node for which the script shall be generated
    :return: The generated script part
    """
    operator_template = env.get_template("operators/" + "function_call.dml.jinja")
    import_template = env.get_template("module_import.dml.jinja")

    if len(list(node.output)) != 1:
        raise Exception("Function call needs output")

    if len(node.attribute) != 0:
        raise Exception("Attributes not supported for this generator")

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
    return GeneratedScriptPart(imports=[import_render], dml_script=node_render)


def gen_2input_1output_operator(env: jinja2.environment.Environment, graph: onnx.GraphProto,
                                node: onnx.NodeProto) -> GeneratedScriptPart:
    """
    Generates simple operator calls like 'z = x + y' which have two inputs (left and right) and one output.
    :param env: Jinja environment to load the template files
    :param graph: the onnx-graph for which the script shall be generated
    :param node: the onnx-node for which the script shall be generated
    :return: The generated script part
    """
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
    return GeneratedScriptPart(node_render)


def gen_1input_1output_mat_operator(env: jinja2.environment.Environment, graph: onnx.GraphProto,
                                    node: onnx.NodeProto) -> GeneratedScriptPart:
    """
    Generates simple operators like 'y = -x' which have one input and one output.
    :param env:  Jinja environment to load the template files
    :param graph: the onnx-graph for which the script shall be generated
    :param node: the onnx-node for which the script shall be generated
    :return: The generated script part
    """
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
    return GeneratedScriptPart(dml_script=node_render)


def gen_dropout_call(env: jinja2.environment.Environment, graph: onnx.GraphProto,
                     node: onnx.NodeProto) -> GeneratedScriptPart:
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
    return GeneratedScriptPart(imports=[import_render], dml_script=node_render)


def __compute_pad(auto_pad: str, Hf: int, Wf: int, strides: [int], pads: [int], Hin: int, Win: int):
    strideh = strides[0]
    stridew = strides[1]

    if auto_pad == "NOTSET":
        padh = pads[0]
        padw = pads[1]
        if pads[0] != pads[2] or pads[1] != pads[3]:
            raise Exception("Only support symmetric pads")
    elif auto_pad == "SAME_UPPER" or "SAME_LOWER":
        # pad such that output size matches input
        padh = (Hin * (strideh - 1) + Hf - strideh) / 2
        padw = (Win * (stridew - 1) + Wf - stridew) / 2
    elif auto_pad == "VALID":
        # no padding
        padh = 0
        padw = 0
    else:
        raise Exception("Invalid auto_pad value")

    return padh, padw


def gen_maxpool_call(env: jinja2.environment.Environment, graph: onnx.GraphProto,
                     node: onnx.NodeProto) -> GeneratedScriptPart:
    operator_template = env.get_template("operators/" + "function_call.dml.jinja")
    import_template = env.get_template("module_import.dml.jinja")

    function_namespace = "maxpool_layer"
    function_name = "forward"
    path = "/nn/layers/max_pool2d.dml"

    #    * Inputs:
    #    *  - X: Inputs, of shape (N, C*Hin*Win).
    #    *  - C: Number of input channels (dimensionality of input depth).
    #    *  - Hin: Input height.
    #    *  - Win: Input width.
    #    *  - Hf: Filter height.
    #    *  - Wf: Filter width.
    #    *  - strideh: Stride over height.
    #    *  - stridew: Stride over width.
    #    *  - padh: Padding for top and bottom sides.
    #    *      A typical value is 0.
    #    *  - padw: Padding for left and right sides.
    #    *      A typical value is 0.
    #    *
    #    * Outputs:
    #    *  - out: Outputs, of shape (N, C*Hout*Wout).
    #    *  - Hout: Output height.
    #    *  - Wout: Output width.

    if len(node.input) != 1:
        raise Exception("Invalid number of inputs")
    if len(node.output) < 1 or len(node.output) > 2:
        raise Exception("Invalid number of outputs")

    # Inputs
    x = onnx_helper.get_value_info(graph, node.input[0])
    # dimensions are (N x C x H x W), where N is the batch size, C is the number of channels,
    # and H and W are the height and the width
    x_shape = onnx_helper.get_valueinfo_dimensions(x)
    if len(x_shape) > 4:
        raise NotImplementedError("Currently only MaxPool-2D supported")

    batch_size = x_shape[0]  # TODO: currently not used
    C = x_shape[1]
    Hin = x_shape[2]
    Win = x_shape[3]

    # Attributes
    auto_pad = "NOTSET"
    ceil_mode = 0
    dilations = [1, 1]
    kernel_shape = None
    pads = [0, 0, 0, 0]
    storage_order = 0
    strides = [1, 1]
    for attribute in node.attribute:
        if attribute.name == "auto_pad":
            auto_pad = attribute.strings[0]
        elif attribute.name == "ceil_mode":
            ceil_mode = attribute.ints[0]
            raise NotImplementedError("Currently no support for ceil_mode")
        elif attribute.name == "dilations":
            raise NotImplementedError
        elif attribute.name == "kernel_shape":
            kernel_shape = attribute.ints
        elif attribute.name == "pads":
            pads = attribute.ints
        elif attribute.name == "storage_order":
            raise NotImplementedError("Currently no support for storage_order")
        elif attribute.name == "strides":
            strides = attribute.ints
        else:
            raise Exception("Invalid Attribute")

    if kernel_shape is None:
        raise Exception("kernel_shape attribute is required")

    Hf = kernel_shape[0]
    Wf = kernel_shape[1]
    strideh = strides[0]
    stridew = strides[1]
    padh, padw = __compute_pad(auto_pad, Hf, Wf, strides, pads, Hin, Win)

    # Create render
    node_render = operator_template.render(
        function_namespace=function_namespace,
        function=function_name,
        arguments=[x.name, C, Hin, Win, Hf, Wf, strideh, stridew, padh, padw],
        outputs=list(node.output),
        doc_string=node.doc_string
    )

    import_render = import_template.render(
        path=path,
        name=function_namespace
    )

    return GeneratedScriptPart(imports=[import_render], dml_script=node_render)


def gen_conv_call(env: jinja2.environment.Environment, graph: onnx.GraphProto, node: onnx.NodeProto) \
        -> GeneratedScriptPart:
    operator_template = env.get_template("operators/" + "function_call.dml.jinja")
    import_template = env.get_template("module_import.dml.jinja")

    function_namespace = "conv_layer"
    function_name = "forward"
    path = "/nn/layers/conv2d.dml"

    #    * Inputs:
    #    *  - X: Inputs, of shape (N, C*Hin*Win).
    #    *  - W: Weights, of shape (F, C*Hf*Wf).
    #    *  - b: Biases, of shape (F, 1).
    #    *  - C: Number of input channels (dimensionality of input depth).
    #    *  - Hin: Input height.
    #    *  - Win: Input width.
    #    *  - Hf: Filter height.
    #    *  - Wf: Filter width.
    #    *  - strideh: Stride over height.
    #    *  - stridew: Stride over width.
    #    *  - padh: Padding for top and bottom sides.
    #    *  - padw: Padding for left and right sides.
    #    *
    #    * Outputs:
    #    *  - out: Outputs, of shape (N, F*Hout*Wout).
    #    *  - Hout: Output height.
    #    *  - Wout: Output width.

    if len(node.input) < 2 or len(node.input) > 3:
        raise Exception("Invalid number of inputs")

    if len(node.output) > 1:
        raise Exception("Invalid number of outputs")

    # Inputs
    x = onnx_helper.get_value_info(graph, node.input[0])
    # size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width
    x_shape = onnx_helper.get_valueinfo_dimensions(x)
    if len(x_shape) > 4:
        raise NotImplementedError("Currently only Conv-2D supported")
    batch_size = x_shape[0]  # TODO: Batch size unused?
    C = x_shape[1]
    Hin = x_shape[2]
    Win = x_shape[3]

    w = onnx_helper.get_value_info(graph, node.input[1])
    W_shape = onnx_helper.get_valueinfo_dimensions(w)
    M = W_shape[0]
    C_group = W_shape[1]  # TODO Channels/group unused?
    Hf = W_shape[2]
    Wf = W_shape[3]

    bias = None
    bias_initializer_render = ""
    if len(node.input) == 2:
        # Generate 0-bias if no bias given
        generated_bias_identifier = "gen_bias"
        while onnx_helper.get_value_info(graph, generated_bias_identifier) is not None:
            # add random number to create unique identifier if already exists
            generated_bias_identifier += str(randint())

        bias_init_template = env.get_template("matrix_initialize.dml.jinja")
        bias_initializer_render = bias_init_template.render(
            identifier_name=generated_bias_identifier,
            initializer_values=[0] * M,
            rows=M,
            cols=1
        )
        bias = generated_bias_identifier
    elif len(node.input) == 3:
        bias = node.input[3]

    # Attributes
    auto_pad = "NOTSET"
    dilations = [1, 1]
    group = 1
    pads = [0, 0, 0, 0]
    strides = [1, 1]
    for attribute in node.attribute:
        if attribute.name == "auto_pad":
            auto_pad = attribute.strings[0]
        elif attribute.name == "dilations":
            raise NotImplementedError
        elif attribute.name == "group":
            group = attribute.ints[0]
        elif attribute.name == "kernel_shape":
            kernel_shape = attribute.ints
            if kernel_shape[0] != Hf or kernel_shape[1] != Wf:
                raise Exception("Invalid kernel shape")
        elif attribute.name == "pads":
            pads = attribute.ints
        elif attribute.name == "strides":
            strides = attribute.ints
        else:
            raise Exception("Invalid Attribute")

    strideh = strides[0]
    stridew = strides[1]
    padh, padw = __compute_pad(auto_pad, Hf, Wf, strides, pads, Hin, Win)

    node_render = operator_template.render(
        function_namespace=function_namespace,
        function=function_name,
        arguments=[x.name, w.name, bias, C, Hin, Win, Hf, Wf, strideh, stridew, padh, padw],
        outputs=list(node.output),
        doc_string=node.doc_string
    )

    import_render = import_template.render(
        path=path,
        name=function_namespace
    )

    return GeneratedScriptPart(imports=[import_render], dml_script=bias_initializer_render + "\n" + node_render)


def gen_if_call(env: jinja2.environment.Environment, graph: onnx.GraphProto, node: onnx.NodeProto) \
        -> GeneratedScriptPart:
    operator_template = env.get_template("operators/if_operator.dml.jinja")
    function_call_template = env.get_template("operators/function_call.dml.jinja")

    if len(node.input) != 1:
        raise Exception("Wrong number of inputs")
    if len(node.attribute) != 2:
        raise Exception("Wrong number of attributes")
    if node.attribute[0].name != "else_branch" or node.attribute[1].name != "then_branch":
        raise Exception("Wrong attributes")

    else_graph = node.attribute[0].g
    then_graph = node.attribute[1].g

    else_call = function_call_template.render(
        doc_string="",
        function_namespace="",
        function=util.generate_function_name(else_graph.name),
        arguments=[i.name for i in list(else_graph.input)],
        outputs=[o.name for o in list(else_graph.output)],
    )

    then_call = function_call_template.render(
        doc_string="",
        function_namespace="",
        function=util.generate_function_name(then_graph.name),
        arguments=[i.name for i in list(then_graph.input)],
        outputs=[o.name for o in list(then_graph.output)],
    )

    sub_graphs = [else_graph, then_graph]

    node_render = operator_template.render(
        cond=node.input[0],
        then_function_call=then_call,
        else_function_call=else_call
    )

    return GeneratedScriptPart(dml_script=node_render, sub_graphs=sub_graphs)
