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
from systemds.utils.converters import matrix_block_to_numpy
from systemds.script_building.script import DMLScript
from systemds.script_building.dag import OutputType, DAGNode


if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from systemds.context import SystemDSContext


class OperationNode(DAGNode):
    """A Node representing an operation in SystemDS"""
    shape: Optional[Tuple[int]]
    _result_var: Optional[Union[float, np.array]]
    _lineage_trace: Optional[str]
    _script: Optional[DMLScript]
    _output_types: Optional[Iterable[VALID_INPUT_TYPES]]

    def __init__(self, sds_context: 'SystemDSContext', operation: str,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None,
                 named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
                 output_type: OutputType = OutputType.MATRIX,
                 is_python_local_data: bool = False,
                 shape: Tuple[int] = (),
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
        self.shape = shape
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

    def compute(self, verbose: bool = False, lineage: bool = False) -> \
            Union[float, np.array, Tuple[Union[float, np.array], str]]:

        if self._result_var is None or self._lineage_trace is None:
            self._script = DMLScript(self.sds_context)
            self._script.build_code(self)
            if verbose:
                print("SCRIPT:")
                print(self._script.dml_script)

            if lineage:
                result_variables, self._lineage_trace = self._script.execute(
                    lineage)
            else:
                result_variables = self._script.execute(lineage)

            if self.output_type == OutputType.DOUBLE:
                self._result_var = result_variables.getDouble(
                    self._script.out_var_name[0])
            elif self.output_type == OutputType.MATRIX:
                self._result_var = matrix_block_to_numpy(self.sds_context.java_gateway.jvm,
                                                         result_variables.getMatrixBlock(self._script.out_var_name[0]))
            elif self.output_type == OutputType.LIST:
                self._result_var = []
                for idx, v in enumerate(self._script.out_var_name):
                    if(self._output_types == None):
                        self._result_var.append(matrix_block_to_numpy(self.sds_context.java_gateway.jvm,
                                                                      result_variables.getMatrixBlock(v)))
                    elif(self._output_types[idx] == OutputType.MATRIX):
                        self._result_var.append(matrix_block_to_numpy(self.sds_context.java_gateway.jvm,
                                                                      result_variables.getMatrixBlock(v)))
                    else:
                        self._result_var.append(result_variables.getDouble(
                            self._script.out_var_name[idx]))
        if verbose:
            for x in self.sds_context.get_stdout():
                print(x)
            for y in self.sds_context.get_stderr():
                print(y)

        if lineage:
            return self._result_var, self._lineage_trace
        else:
            return self._result_var

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
        elif self.output_type == OutputType.ASSIGN:
            return f'{var_name}={self.operation};'
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

    def __add__(self, other: VALID_ARITHMETIC_TYPES) -> 'OperationNode':
        return OperationNode(self.sds_context, '+', [self, other], shape=self.shape)

    def __sub__(self, other: VALID_ARITHMETIC_TYPES) -> 'OperationNode':
        return OperationNode(self.sds_context, '-', [self, other], shape=self.shape)

    def __mul__(self, other: VALID_ARITHMETIC_TYPES) -> 'OperationNode':
        return OperationNode(self.sds_context, '*', [self, other], shape=self.shape)

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES) -> 'OperationNode':
        return OperationNode(self.sds_context, '/', [self, other], shape=self.shape)

    def __floordiv__(self, other: VALID_ARITHMETIC_TYPES) -> 'OperationNode':
        return OperationNode(self.sds_context, '//', [self, other], shape=self.shape)

    def __lt__(self, other) -> 'OperationNode':
        return OperationNode(self.sds_context, '<', [self, other], shape=self.shape)

    def __le__(self, other) -> 'OperationNode':
        return OperationNode(self.sds_context, '<=', [self, other], shape=self.shape)

    def __gt__(self, other) -> 'OperationNode':
        return OperationNode(self.sds_context, '>', [self, other], shape=self.shape)

    def __ge__(self, other) -> 'OperationNode':
        return OperationNode(self.sds_context, '>=', [self, other], shape=self.shape)

    def __eq__(self, other) -> 'OperationNode':
        return OperationNode(self.sds_context, '==', [self, other], shape=self.shape)

    def __ne__(self, other) -> 'OperationNode':
        return OperationNode(self.sds_context, '!=', [self, other], shape=self.shape)

    def __matmul__(self, other: 'OperationNode') -> 'OperationNode':
        return OperationNode(self.sds_context, '%*%', [self, other], shape=(self.shape[0], other.shape[0]))

    def sum(self, axis: int = None) -> 'OperationNode':
        """Calculate sum of matrix.

        :param axis: can be 0 or 1 to do either row or column sums
        :return: `OperationNode` representing operation
        """
        self._check_matrix_op()
        if axis == 0:
            return OperationNode(self.sds_context, 'colSums', [self], shape=(self.shape[1],))
        elif axis == 1:
            return OperationNode(self.sds_context, 'rowSums', [self], shape=(self.shape[0],))
        elif axis is None:
            return OperationNode(self.sds_context, 'sum', [self], output_type=OutputType.DOUBLE)
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def mean(self, axis: int = None) -> 'OperationNode':
        """Calculate mean of matrix.

        :param axis: can be 0 or 1 to do either row or column means
        :return: `OperationNode` representing operation
        """
        self._check_matrix_op()
        if axis == 0:
            return OperationNode(self.sds_context, 'colMeans', [self], shape=(self.shape[1],))
        elif axis == 1:
            return OperationNode(self.sds_context, 'rowMeans', [self], shape=(self.shape[0],))
        elif axis is None:
            return OperationNode(self.sds_context, 'mean', [self], output_type=OutputType.DOUBLE)
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def var(self, axis: int = None) -> 'OperationNode':
        """Calculate variance of matrix.

        :param axis: can be 0 or 1 to do either row or column vars
        :return: `OperationNode` representing operation
        """
        self._check_matrix_op()
        if axis == 0:
            return OperationNode(self.sds_context, 'colVars', [self], shape=(self.shape[1],))
        elif axis == 1:
            return OperationNode(self.sds_context, 'rowVars', [self], shape=(self.shape[0],))
        elif axis is None:
            return OperationNode(self.sds_context, 'var', [self], output_type=OutputType.DOUBLE)
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def abs(self) -> 'OperationNode':
        """Calculate absolute.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'abs', [self], shape = self.shape)

    def sin(self) -> 'OperationNode':
        """Calculate sin.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'sin', [self], shape = self.shape)

    def cos(self) -> 'OperationNode':
        """Calculate cos.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'cos', [self], shape = self.shape)

    def tan(self) -> 'OperationNode':
        """Calculate tan.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'tan', [self], shape = self.shape)

    def asin(self) -> 'OperationNode':
        """Calculate arcsin.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'asin', [self], shape = self.shape)

    def acos(self) -> 'OperationNode':
        """Calculate arccos.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'acos', [self], shape = self.shape)

    def atan(self) -> 'OperationNode':
        """Calculate arctan.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'atan', [self], shape = self.shape)

    def sinh(self) -> 'OperationNode':
        """Calculate sin.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'sinh', [self], shape = self.shape)

    def cosh(self) -> 'OperationNode':
        """Calculate cos.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'cosh', [self], shape = self.shape)

    def tanh(self) -> 'OperationNode':
        """Calculate tan.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'tanh', [self], shape = self.shape)

    def moment(self, moment, weights: DAGNode = None) -> 'OperationNode':
        # TODO write tests
        self._check_matrix_op()
        unnamed_inputs = [self]
        if weights is not None:
            unnamed_inputs.append(weights)
        unnamed_inputs.append(moment)
        return OperationNode(self.sds_context, 'moment', unnamed_inputs, output_type=OutputType.DOUBLE)

    def write(self, destination: str, format:str = "binary", **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'OperationNode':
        """ Write input to disk. 
        The written format is easily read by SystemDSContext.read(). 
        There is no return on write.

        :param destination: The location which the file is stored. Defaulting to HDFS paths if available.
        :param format: The format which the file is saved in. Default is binary to improve SystemDS reading times.
        :param kwargs: Contains multiple extra specific arguments, can be seen at http://apache.github.io/systemds/site/dml-language-reference#readwrite-built-in-functions
        """
        unnamed_inputs = [self, f'"{destination}"']
        named_parameters = {"format":f'"{format}"'}
        named_parameters.update(kwargs)
        return OperationNode(self.sds_context, 'write', unnamed_inputs, named_parameters, output_type= OutputType.NONE)

    def to_string(self, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'OperationNode':
        """ Converts the input to a string representation.
        :return: `OperationNode` containing the string.
        """
        return OperationNode(self.sds_context, 'toString', [self], kwargs, output_type= OutputType.SCALAR)

    def print(self, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'OperationNode':
        """ Prints the given Operation Node.
        There is no return on calling.
        To get the returned string look at the stdout of SystemDSContext.
        """
        return OperationNode(self.sds_context, 'print', [self], kwargs, output_type= OutputType.NONE)

    def rev(self) -> 'OperationNode':
        """ Reverses the rows in a matrix

        :return: the OperationNode representing this operation
        """

        self._check_matrix_op()
        return OperationNode(self.sds_context, 'rev', [self])

    def order(self, by: int = 1, decreasing: bool = False,
              index_return: bool = False) -> 'OperationNode':
        """ Sort by a column of the matrix X in increasing/decreasing order and returns either the index or data

        :param by: sort matrix by this column number
        :param decreasing: If true the matrix will be sorted in decreasing order
        :param index_return: If true, the index numbers will be returned
        :return: the OperationNode representing this operation
        """

        self._check_matrix_op()

        cols = self._np_array.shape[1]
        if by > cols:
            raise IndexError(
                "Index {i} is out of bounds for axis 1 with size {c}".format(i=by, c=cols))

        named_input_nodes = {'target': self, 'by': by, 'decreasing': str(decreasing).upper(),
                             'index.return': str(index_return).upper()}

        return OperationNode(self.sds_context, 'order', [], named_input_nodes=named_input_nodes)

    def t(self) -> 'OperationNode':
        """ Transposes the input matrix

        :return: the OperationNode representing this operation
        """

        self._check_matrix_op()
        return OperationNode(self.sds_context, 't', [self])

    def cholesky(self, safe: bool = False) -> 'OperationNode':
        """ Computes the Cholesky decomposition of a symmetric, positive definite matrix

        :param safe: default value is False, if flag is True additional checks to ensure
            that the matrix is symmetric positive definite are applied, if False, checks will be skipped
        :return: the OperationNode representing this operation
        """

        self._check_matrix_op()
        # check square dimension
        if self.shape[0] != self.shape[1]:
            raise ValueError("Last 2 dimensions of the array must be square")

        if safe:
            # check if mat is positive definite
            if not np.all(np.linalg.eigvals(self._np_array) > 0):
                raise ValueError("Matrix is not positive definite")

            # check if mat is symmetric
            if not np.allclose(self._np_array, self._np_array.transpose()):
                raise ValueError("Matrix is not symmetric")

        return OperationNode(self.sds_context, 'cholesky', [self])

    def to_one_hot(self, num_classes: int) -> 'OperationNode':
        """ OneHot encode the matrix.

        It is assumed that there is only one column to encode, and all values are whole numbers > 0

        :param num_classes: The number of classes to encode into. max value contained in the matrix must be <= num_classes
        :return: The OperationNode containing the oneHotEncoded values
        """
        
        self._check_matrix_op()
        if len(self.shape) != 1:
            raise ValueError(
                "Only Matrixes  with a single column or row is valid in One Hot, " + str(self.shape) + " is invalid")

        if num_classes < 2:
            raise ValueError("Number of classes should be larger than 1")

        named_input_nodes = {"X": self, "numClasses": num_classes}
        return OperationNode(self.sds_context, 'toOneHot', named_input_nodes=named_input_nodes, shape=(self.shape[0], num_classes))