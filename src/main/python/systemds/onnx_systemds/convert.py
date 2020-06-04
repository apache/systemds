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

import argparse
import os.path
import systemds.onnx_systemds.onnx_helper as onnx_helper
from systemds.onnx_systemds import render


def init_argparse() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description="Convert onnx models into dml scripts")
    arg_parser.add_argument("input", type=str)
    arg_parser.add_argument("-o", "--output", type=str,
                            help="output file", required=False)
    return arg_parser


def onnx2systemds(input_onnx_file: str, output_dml_file: str = None) -> None:
    """
    Loads the model from the input file and generates a dml file.

    :param input_onnx_file: the onnx input file
    :param output_dml_file: (optional) the dml output file,
        if this parameter is not given the output file will have the same name as the input file
    """
    if not os.path.isfile(input_onnx_file):
        raise Exception("Invalid input-file: " + str(input_onnx_file))

    if not output_dml_file:
        output_dml_file = os.path.splitext(os.path.basename(input_onnx_file))[0] + ".dml"

    model = onnx_helper.load_model(input_onnx_file)
    render.gen_script(model, output_dml_file)


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    onnx2systemds(input_file, output_file)
