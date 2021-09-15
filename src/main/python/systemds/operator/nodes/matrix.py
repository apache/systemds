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

__all__ = ["Matrix"]

import os
from typing import (TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple,
                    Union)

import numpy as np
from py4j.java_gateway import JavaObject, JVMView
from systemds.operator import OperationNode, Scalar
from systemds.script_building.dag import OutputType
from systemds.utils.consts import (BINARY_OPERATIONS, VALID_ARITHMETIC_TYPES,
                                   VALID_INPUT_TYPES)
from systemds.utils.converters import (matrix_block_to_numpy,
                                       numpy_to_matrix_block)
from systemds.utils.helpers import get_slice_string


class Matrix(OperationNode):
    _np_array: np.array

    def __init__(self, sds_context: 'SystemDSContext', operation: str,
                 unnamed_input_nodes: Union[str,
                                            Iterable[VALID_INPUT_TYPES]] = None,
                 named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
                 local_data: np.array = None, brackets: bool = False) -> 'Matrix':

        is_python_local_data = False
        if local_data is not None:
            self._np_array = local_data
            is_python_local_data = True
        else:
            self._np_array = None

        super().__init__(sds_context, operation, unnamed_input_nodes,
                         named_input_nodes, OutputType.MATRIX, is_python_local_data, brackets)

    def pass_python_data_to_prepared_script(self, sds, var_name: str, prepared_script: JavaObject) -> None:
        assert self.is_python_local_data, 'Can only pass data to prepared script if it is python local!'
        if self._is_numpy():
            prepared_script.setMatrix(var_name, numpy_to_matrix_block(
                sds, self._np_array), True)  # True for reuse

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        code_line = super().code_line(var_name, unnamed_input_vars, named_input_vars)
        if self._is_numpy():
            code_line = code_line.format(file_name=var_name)
        return code_line

    def compute(self, verbose: bool = False, lineage: bool = False) -> Union[np.array]:
        if self._is_numpy():
            if verbose:
                print('[Numpy Array - No Compilation necessary]')
            return self._np_array
        else:
            return super().compute(verbose, lineage)

    def _parse_output_result_variables(self, result_variables):
        return matrix_block_to_numpy(self.sds_context.java_gateway.jvm,
                                     result_variables.getMatrixBlock(self._script.out_var_name[0]))

    def _is_numpy(self) -> bool:
        return self._np_array is not None

    def __add__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.sds_context, '+', [self, other])

    # Left hand side
    def __radd__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.sds_context, '+', [other, self])

    def __sub__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.sds_context, '-', [self, other])

    # Left hand side
    def __rsub__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.sds_context, '-', [other, self])

    def __mul__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.sds_context, '*', [self, other])

    def __rmul__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.sds_context, '*', [other, self])

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.sds_context, '/', [self, other])

    def __rtruediv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.sds_context, '/', [other, self])

    def __floordiv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.sds_context, '//', [self, other])

    def __rfloordiv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Matrix':
        return Matrix(self.sds_context, '//', [other, self])

    def __lt__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '<', [self, other])

    def __rlt__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '<', [other, self])

    def __le__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '<=', [self, other])

    def __rle__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '<=', [other, self])

    def __gt__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '>', [self, other])

    def __rgt__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '>', [other, self])

    def __ge__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '>=', [self, other])

    def __rge__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '>=', [other, self])

    def __eq__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '==', [self, other])

    def __req__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '==', [other, self])

    def __ne__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '!=', [self, other])

    def __rne__(self, other) -> 'Matrix':
        return Matrix(self.sds_context, '!=', [other, self])

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        return Matrix(self.sds_context, '%*%', [self, other])

    def __getitem__(self, i):
        sliceIns = get_slice_string(i)
        return Matrix(self.sds_context, '', [self, sliceIns], brackets=True)

    def sum(self, axis: int = None) -> 'OperationNode':
        """Calculate sum of matrix.

        :param axis: can be 0 or 1 to do either row or column sums
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, 'colSums', [self])
        elif axis == 1:
            return Matrix(self.sds_context, 'rowSums', [self])
        elif axis is None:
            return Scalar(self.sds_context, 'sum', [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def mean(self, axis: int = None) -> 'OperationNode':
        """Calculate mean of matrix.

        :param axis: can be 0 or 1 to do either row or column means
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, 'colMeans', [self])
        elif axis == 1:
            return Matrix(self.sds_context, 'rowMeans', [self])
        elif axis is None:
            return Scalar(self.sds_context, 'mean', [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def max(self, axis: int = None) -> 'OperationNode':
        """Calculate max of matrix.

        :param axis: can be 0 or 1 to do either row or column aggregation
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, 'colMaxs', [self])
        elif axis == 1:
            return Matrix(self.sds_context, 'rowMaxs', [self])
        elif axis is None:
            return Scalar(self.sds_context, 'max', [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def min(self, axis: int = None) -> 'OperationNode':
        """Calculate max of matrix.

        :param axis: can be 0 or 1 to do either row or column aggregation
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, 'colMins', [self])
        elif axis == 1:
            return Matrix(self.sds_context, 'rowMins', [self])
        elif axis is None:
            return Scalar(self.sds_context, 'min', [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def var(self, axis: int = None) -> 'OperationNode':
        """Calculate variance of matrix.

        :param axis: can be 0 or 1 to do either row or column vars
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, 'colVars', [self])
        elif axis == 1:
            return Matrix(self.sds_context, 'rowVars', [self])
        elif axis is None:
            return Scalar(self.sds_context, 'var', [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def abs(self) -> 'Matrix':
        """Calculate absolute.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, 'abs', [self])

    def sin(self) -> 'Matrix':
        """Calculate sin.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, 'sin', [self])

    def cos(self) -> 'Matrix':
        """Calculate cos.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, 'cos', [self])

    def tan(self) -> 'Matrix':
        """Calculate tan.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, 'tan', [self])

    def asin(self) -> 'Matrix':
        """Calculate arcsin.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, 'asin', [self])

    def acos(self) -> 'Matrix':
        """Calculate arccos.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, 'acos', [self])

    def atan(self) -> 'Matrix':
        """Calculate arctan.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, 'atan', [self])

    def sinh(self) -> 'Matrix':
        """Calculate sin.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, 'sinh', [self])

    def cosh(self) -> 'Matrix':
        """Calculate cos.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, 'cosh', [self])

    def tanh(self) -> 'Matrix':
        """Calculate tan.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, 'tanh', [self])

    def moment(self, moment: int, weights: OperationNode = None) -> 'Matrix':
        unnamed_inputs = [self]
        if weights is not None:
            unnamed_inputs.append(weights)
        unnamed_inputs.append(moment)
        return Matrix(self.sds_context, 'moment', unnamed_inputs, output_type=OutputType.DOUBLE)

    def cholesky(self, safe: bool = False) -> 'Matrix':
        """ Computes the Cholesky decomposition of a symmetric, positive definite matrix

        :param safe: default value is False, if flag is True additional checks to ensure
            that the matrix is symmetric positive definite are applied, if False, checks will be skipped
        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, 'cholesky', [self])

    def to_one_hot(self, num_classes: int) -> 'Matrix':
        """ OneHot encode the matrix.

        It is assumed that there is only one column to encode, and all values are whole numbers > 0

        :param num_classes: The number of classes to encode into. max value contained in the matrix must be <= num_classes
        :return: The OperationNode containing the oneHotEncoded values
        """
        if num_classes < 2:
            raise ValueError("Number of classes should be larger than 1")

        named_input_nodes = {"X": self, "numClasses": num_classes}
        return Matrix(self.sds_context, 'toOneHot', named_input_nodes=named_input_nodes)

    def rbind(self, other) -> 'Matrix':
        """
        Row-wise matrix concatenation, by concatenating the second matrix as additional rows to the first matrix. 
        :param: The other matrix to bind to the right hand side
        :return: The OperationNode containing the concatenated matrices/frames.
        """
        return Matrix(self.sds_context, "rbind", [self, other])

    def cbind(self, other) -> 'Matrix':
        """
        Column-wise matrix concatenation, by concatenating the second matrix as additional columns to the first matrix. 
        :param: The other matrix to bind to the right hand side.
        :return: The OperationNode containing the concatenated matrices/frames.
        """
        return Matrix(self.sds_context, "cbind", [self, other])

    def t(self) -> 'Matrix':
        """ Transposes the input

        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, 't', [self])

    def order(self, by: int = 1, decreasing: bool = False,
              index_return: bool = False) -> 'Matrix':
        """ Sort by a column of the matrix X in increasing/decreasing order and returns either the index or data

        :param by: sort matrix by this column number
        :param decreasing: If true the matrix will be sorted in decreasing order
        :param index_return: If true, the index numbers will be returned
        :return: the OperationNode representing this operation
        """

        named_input_nodes = {'target': self, 'by': by, 'decreasing': str(decreasing).upper(),
                             'index.return': str(index_return).upper()}

        return Matrix(self.sds_context, 'order', [], named_input_nodes=named_input_nodes)

    def to_string(self, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'Scalar':
        """ Converts the input to a string representation.
        :return: `Scalar` containing the string.
        """
        return Scalar(self.sds_context, 'toString', [self], kwargs, output_type=OutputType.STRING)

    def rev(self) -> 'Matrix':
        """ Reverses the rows

        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, 'rev', [self])

    def round(self) -> 'Matrix':
        """ round all values to nearest natural number

        :return: The Matrix representing the result of this operation
        """
        return Matrix(self.sds_context, "round", [self])

    def replace(self, pattern: VALID_INPUT_TYPES, replacement: VALID_INPUT_TYPES) -> 'Matrix':
        """
        Replace all values with replacement value
        """
        return Matrix(self.sds_context, "replace", named_input_nodes={"target": self, "pattern": pattern, "replacement": replacement})

    def __str__(self):
        return "MatrixNode"
