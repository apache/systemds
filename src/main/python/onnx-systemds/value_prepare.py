# Classes and functions in this file take the ProtoBuf destructure and prepare values / perform checks needed for
# rendering the dml script if however the information can simply be read from the ProtoBuf structure such a class is
# not needed
import onnx


class OnnxValue:
    def __init__(self, value_info: onnx.ValueInfoProto, initializer: onnx.TensorProto = None):

        # TODO: check prob not correct
        if value_info.type.tensor_type.elem_type != 1:
            raise NotImplementedError("Only support Tensor Types")

        self.shape = []
        shape_dimensions = value_info.type.tensor_type.shape.dim
        if len(shape_dimensions) > 2:
            # TODO: might want to add support for that
            raise NotImplementedError("Only support up to 2 dimensions")

        for dim in shape_dimensions:
            # TODO: shapes with no value but instead name -> support?
            if len(dim.dim_param) != 0:
                raise NotImplementedError("Only support dim_value")
            self.shape.append(dim.dim_value)

        self.identifier_name = value_info.name
        self.description = value_info.doc_string
        self.data_type = "matrix"
        self.value_type = "double"  # TODO: other types + translation
        self.initializer = None

        # TODO: initializers

        # TODO: deal with unsuported data types
        type_translation = {
            1: "double",  # float
            2: "int",  # uint8_t
            3: "int",  # int8_t
            4: "int",  # uint16_t
            5: "int",  # int16_t
            6: "int",  # int32_t
            7: "int",  # int64_t
            8: "string",
            9: "bool",

            10: "double",  # float16,
            11: "double",
            12: "int",  # uint32
            13: "int",  # uint64

            # TODO: deal with unsuported data types
            14: "COMPLEX64",
            15: "COMPLEX128",
            16: "BFLOAT16"
        }

        if initializer:
            self.initializer_values = list(initializer.float_data)


def prepare_function_inputs(inputs: [onnx.ValueInfoProto]) -> [OnnxValue]:
    return [OnnxValue(i) for i in inputs]


def prepare_initialized_inputs(inputs: [(onnx.ValueInfoProto, onnx.TensorProto)]) -> [OnnxValue]:
    return [OnnxValue(info, init) for info, init in inputs]


def prepare_function_outputs(outputs: [onnx.ValueInfoProto]) -> [OnnxValue]:
    return [OnnxValue(o) for o in outputs]

