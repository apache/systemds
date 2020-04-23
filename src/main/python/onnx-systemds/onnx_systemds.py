import argparse
import os.path
import ntpath
import onnx_model
import render


def init_argparse() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description="Convert onnx models into dml scripts")
    arg_parser.add_argument("input", type=str)
    arg_parser.add_argument("-o", "--output", type=str,
                            help="output file", required=False)
    return arg_parser


def onnx2systemds(input_onnx_file: str, output_dml_file: str = None) -> None:
    """
    Generates the dml script from a onnx file
    :param input_onnx_file: the onnx input file
    :param output_dml_file: the dml output file,
        if this parameter is not given or is None the output file will be the input file with dml file-extension
    """
    if not os.path.isfile(input_onnx_file):
        raise Exception("Invalid input-file: " + str(input_onnx_file))

    if not output_dml_file:
        output_dml_file = os.path.splitext(ntpath.basename(input_onnx_file))[0] + ".dml"

    graph = onnx_model.OnnxModel(input_onnx_file)
    render.write_script(graph, output_dml_file)


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    onnx2systemds(input_file, output_file)
