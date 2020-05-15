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
import subprocess
import unittest

from systemds.onnx_systemds.convert import onnx2systemds
from systemds.onnx_systemds.util import resolve_systemds_root


def invoke_systemds(input_file: str, args: [str] = None) -> int:
    """
    Runs systemds by running the script in $SYSTEMDS_ROOT/bin/systemds with the provided input_file,
    will fail if environment variable SYSTEMDS_ROOT is not set.

    Furthermore this if the script is successfully this method will print lines from Log4J if they are:
    WARN or ERROR.

    :param input_file: the dml script to run
    :param args: additional arguments if needed
    :return: the return-code of systemds
    """
    if args is None:
        args = []

    systemds_root_path = resolve_systemds_root()

    try:
        realpath_input = os.path.relpath(input_file, os.getcwd())
        res = subprocess.run([systemds_root_path + "/bin/systemds", realpath_input] + args,
                             check=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             timeout=10000)
    except subprocess.CalledProcessError as systemds_error:
        print("SYSTEMDS FAILED!")
        print("error code: " + str(systemds_error.returncode))
        print("Stdout:")
        print(systemds_error.output.decode("utf-8"))
        print("Stderr:")
        print(systemds_error.stderr.decode("utf-8"))
        return systemds_error.returncode

    stderr = res.stderr.decode("utf-8")
    if len(stderr) != 0:
        lines = [l for l in stderr.split("\n") if ("WARN" in l or "ERROR" in l)]
        if len(lines) != 0:
            print("No exception but output but warnings and errors:")
            [print(l) for l in lines]

    return res.returncode


def run_and_compare_output(name: str, test_case: unittest.TestCase) -> None:
    """
    Converts the onnx-model to dml, runs systemds with the dml-wrapper and compares the resulting output
     with the reference output.
    :param name: The name of the test-case (also used for finding onnx-model, dml-wrapper and reference output)
    :param test_case: The testcase
    """
    onnx2systemds("tests/onnx_systemds/test_models/" + name + ".onnx", "tests/onnx_systemds/dml_output/" + name + ".dml")
    ret = invoke_systemds("tests/onnx_systemds/dml_wrapper/" + name + "_wrapper.dml")
    test_case.assertEqual(ret, 0, "systemds failed")

    # We read the file content such that pytest can present the actual difference between the files
    with open("tests/onnx_systemds/output_reference/" + name + "_reference.out") as reference_file:
        reference_content = reference_file.read()

    with open("tests/onnx_systemds/output_test/" + name + ".out") as output_file:
        test_content = output_file.read()

    test_case.assertEqual(
        test_content,
        reference_content,
        "generated output differed from reference output"
    )
