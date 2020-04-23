import filecmp
import io
import os
import subprocess
import unittest


from onnx_systemds import onnx2systemds


def invoke_systemds(input_file: str, args: [str] = None) -> int:
    if args is None:
        args = []
    SYSTEMDS_ROOT_PATH = ""
    try:
        SYSTEMDS_ROOT_PATH = os.environ['SYSTEMDS_ROOT']
    except KeyError as error:
        print("ERROR environment variable SYSTEMDS_ROOT_PATH not set")
        exit(-1)

    try:
        abspath_input = os.path.abspath(input_file)
        res = subprocess.run([SYSTEMDS_ROOT_PATH + "/bin/systemds.sh", abspath_input] + args,
                             check=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as systemds_error:
        print("SYSTEMDS FAILED!")
        print("error code: " + str(systemds_error.returncode))
        print(str(systemds_error.output))
        return systemds_error.returncode

    return res.returncode


def run_and_compare_output(name: str, test_case: unittest.TestCase) -> None:
    onnx2systemds("test_models/" + name + ".onnx", "dml_output/" + name + ".dml")
    ret = invoke_systemds("dml_wrapper/" + name + "_wrapper.dml")
    test_case.assertEqual(ret, 0, "systemds exit code was not 0")

    # We read the file content such that pytest can present the actual difference between the files
    with open("output_reference/" + name + "_reference.out") as reference_file:
        reference_content = reference_file.read()

    with open("output_test/" + name + ".out") as output_file:
        test_content = output_file.read()

    test_case.assertNotEqual(reference_content, None)
    test_case.assertNotEqual(test_content, None)
    test_case.assertEqual(
        test_content,
        reference_content,
        "generated output differed from reference output"
    )
