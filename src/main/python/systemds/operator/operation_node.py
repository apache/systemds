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

from typing import Union, Optional, Iterable, Dict, Sequence, Tuple, TYPE_CHECKING
from multiprocessing import Process

import numpy as np
from py4j.java_gateway import JVMView, JavaObject

from systemds.utils.consts import VALID_INPUT_TYPES, BINARY_OPERATIONS, VALID_ARITHMETIC_TYPES
from systemds.utils.helpers import create_params_string
from systemds.utils.converters import matrix_block_to_numpy, frame_block_to_pandas
from systemds.script_building.script import DMLScript
from systemds.script_building.dag import OutputType, DAGNode

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from systemds.context import SystemDSContext


class OperationNode(DAGNode):
    """A Node representing an operation in SystemDS"""

    _result_var: Optional[Union[float, np.array]]
    _lineage_trace: Optional[str]
    _script: Optional[DMLScript]
    _output_types: Optional[Iterable[VALID_INPUT_TYPES]]
    _source_node: Optional["DAGNode"]
    
    def __init__(self, sds_context: 'SystemDSContext', operation: str,
                 unnamed_input_nodes: Union[str,
                                            Iterable[VALID_INPUT_TYPES]] = None,
                 named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
                 output_type: OutputType = OutputType.MATRIX,
                 is_python_local_data: bool = False,
                 number_of_outputs=1,
                 output_types: Iterable[OutputType] = None):
                 
        """
        Create general `OperationNode`

        :param sds_context: The SystemDS context for performing the operations
        :param operation: The name of the DML function to execute
        :param unnamed_input_nodes: inputs identified by their position, not name
        :param named_input_nodes: inputs with their respective parameter name
        :param output_type: type of the output in DML (double, matrix etc.)
        :param is_python_local_data: if the data is local in python e.g. Numpy arrays
        :param number_of_outputs: If set to other value than 1 then it is expected
            that this operation node returns multiple values. If set remember to set the output_types value as well.
        :param output_types: The types of output in a multi output scenario.
            Default is None, and means every multi output is a matrix.
        """
        self.sds_context = sds_context
        if unnamed_input_nodes is None:
            unnamed_input_nodes = []
        if named_input_nodes is None:
            named_input_nodes = {}
        self.operation = operation
        self._unnamed_input_nodes = unnamed_input_nodes
        self._named_input_nodes = named_input_nodes
        self._output_type = output_type
        self._is_python_local_data = is_python_local_data
        self._result_var = None
        self._lineage_trace = None
        self._script = None
        self._number_of_outputs = number_of_outputs
        self._output_types = output_types
        self._source_node = None
        self._already_added = False

    def compute(self, verbose: bool = False, lineage: bool = False) -> \
            Union[float, np.array, Tuple[Union[float, np.array], str]]:

        if self._result_var is None or self._lineage_trace is None:
            self._script = DMLScript(self.sds_context)
            self._script.build_code(self)
            if verbose:
                print("SCRIPT:")
                print(self._script.dml_script)

            if lineage:
                result_variables, self._lineage_trace = self._script.execute_with_lineage()
            else:
                result_variables = self._script.execute()

            if result_variables is not None:
                self._result_var = self.__parse_output_result_variables(
                    result_variables)

        if verbose:
            for x in self.sds_context.get_stdout():
                print(x)
            for y in self.sds_context.get_stderr():
                print(y)

        if lineage:
            return self._result_var, self._lineage_trace
        else:
            return self._result_var

    def __parse_output_result_variables(self, result_variables):
        if self.output_type == OutputType.DOUBLE:
            return self.__parse_output_result_double(result_variables, self._script.out_var_name[0])
        elif self.output_type == OutputType.MATRIX:
            return self.__parse_output_result_matrix(result_variables, self._script.out_var_name[0])
        elif self.output_type == OutputType.LIST:
            return self.__parse_output_result_list(result_variables)
        elif self.output_type == OutputType.FRAME:
            return self.__parse_output_result_frame(result_variables, self._script.out_var_name[0])

    def __parse_output_result_double(self, result_variables, var_name):
        return result_variables.getDouble(var_name)

    def __parse_output_result_matrix(self, result_variables, var_name):
        return matrix_block_to_numpy(self.sds_context.java_gateway.jvm,
                                     result_variables.getMatrixBlock(var_name))

    def __parse_output_result_frame(self, result_variables, var_name):
        return frame_block_to_pandas(
            self.sds_context, result_variables.getFrameBlock(var_name)
        )

    def __parse_output_result_list(self, result_variables):
        result_var = []
        for idx, v in enumerate(self._script.out_var_name):
            if(self._output_types == None or self._output_types[idx] == OutputType.MATRIX):
                result_var.append(
                    self.__parse_output_result_matrix(result_variables, v))
            elif self._output_types[idx] == OutputType.FRAME:
                result_var.append(
                    self.__parse_output_result_frame(result_variables, v))

            else:
                result_var.append(result_variables.getDouble(
                    self._script.out_var_name[idx]))
        return result_var

    def get_lineage_trace(self) -> str:
        """Get the lineage trace for this node.

        :return: Lineage trace
        """
        if self._lineage_trace is None:
            self._script = DMLScript(self.sds_context)
            self._script.build_code(self)
            self._lineage_trace = self._script.get_lineage()

        return self._lineage_trace

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        if self.operation in BINARY_OPERATIONS:
            assert len(
                named_input_vars) == 0, 'Named parameters can not be used with binary operations'
            assert len(
                unnamed_input_vars) == 2, 'Binary Operations need exactly two input variables'
            return f'{var_name}={unnamed_input_vars[0]}{self.operation}{unnamed_input_vars[1]}'

        inputs_comma_sep = create_params_string(
            unnamed_input_vars, named_input_vars)

        if self.output_type == OutputType.LIST:
            output = "["
            for idx in range(self._number_of_outputs):
                output += f'{var_name}_{idx},'
            output = output[:-1] + "]"
            return f'{output}={self.operation}({inputs_comma_sep});'
        elif self.output_type == OutputType.NONE:
            return f'{self.operation}({inputs_comma_sep});'
        # elif self.output_type == OutputType.ASSIGN:
        #     return f'{var_name}={self.operation};'
        else:
            return f'{var_name}={self.operation}({inputs_comma_sep});'

    def pass_python_data_to_prepared_script(self, jvm: JVMView, var_name: str, prepared_script: JavaObject) -> None:
        raise NotImplementedError(
            'Operation node has no python local data. Missing implementation in derived class?')

    def _check_matrix_op(self):
        """Perform checks to assure operation is allowed to be performed on data type of this `OperationNode`

        :raise: AssertionError
        """
        assert self.output_type == OutputType.MATRIX, f'{self.operation} only supported for matrices'

    def _check_frame_op(self):
        """Perform checks to assure operation is allowed to be performed on data type of this `OperationNode`

        :raise: AssertionError
        """
        assert self.output_type == OutputType.FRAME, f'{self.operation} only supported for frames'

    def _check_matrix_or_frame_op(self):
        """Perform checks to assure operation is allowed to be performed on data type of this `OperationNode`

        :raise: AssertionError
        """
        assert (
            self.output_type == OutputType.FRAME
            or self.output_type == OutputType.MATRIX
        ), f"{self.operation} only supported for frames or matrices"

    def _check_equal_op_type_as(self, other: "OperationNode"):
        """Perform checks to assure operation is equal to 'other'. Used for rBind and cBind type equality check.

        :raise: AssertionError
        """
        assert (
            self.output_type == other.output_type
        ), f"{self.operation} only supported for Nodes of equal output-type. Got self: {self.output_type} and other: {other.output_type}"

    def _check_other(self, other: "OperationNode", expectedOutputType: OutputType):
        """Perform check to assure other operation has expected output type.

        :raise: AssertionError
        """
        assert other.output_type == expectedOutputType, "not correctly asserted output types expected: " + \
            str(expectedOutputType) + " got " + str(other.output_type)

    def write(self, destination: str, format: str = "binary", **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'OperationNode':
        """ Write input to disk. 
        The written format is easily read by SystemDSContext.read(). 
        There is no return on write.

        :param destination: The location which the file is stored. Defaulting to HDFS paths if available.
        :param format: The format which the file is saved in. Default is binary to improve SystemDS reading times.
        :param kwargs: Contains multiple extra specific arguments, can be seen at http://apache.github.io/systemds/site/dml-language-reference#readwrite-built-in-functions
        """
        unnamed_inputs = [self, f'"{destination}"']
        named_parameters = {"format": f'"{format}"'}
        named_parameters.update(kwargs)
        return OperationNode(self.sds_context, 'write', unnamed_inputs, named_parameters, output_type=OutputType.NONE)

    def print(self, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'OperationNode':
        """ Prints the given Operation Node.
        There is no return on calling.
        To get the returned string look at the stdout of SystemDSContext.
        """
        return OperationNode(self.sds_context, 'print', [self], kwargs, output_type=OutputType.NONE)

    def rev(self) -> 'OperationNode':
        """ Reverses the rows

        :return: the OperationNode representing this operation
        """
        return OperationNode(self.sds_context, 'rev', [self])

    def to_string(self, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'OperationNode':
        """ Converts the input to a string representation.
        :return: `Scalar` containing the string.
        """
        return OperationNode(self.sds_context, 'toString', [self], kwargs, output_type=OutputType.STRING)