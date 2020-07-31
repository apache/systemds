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

import os
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from py4j.java_gateway import JavaObject, JVMView
from systemds.context import SystemDSContext
from systemds.operator import OperationNode
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.utils.converters import numpy_to_matrix_block

# TODO maybe instead of having a new class we could have a function `matrix` instead, adding behavior to
#  `OperationNode` would be necessary


class Matrix(OperationNode):
    _np_array: Optional[np.array]

    def __init__(self, sds_context: 'SystemDSContext', mat: Union[np.array, os.PathLike],
                 *args: Sequence[VALID_INPUT_TYPES],
                 **kwargs: Dict[str, VALID_INPUT_TYPES]) -> None:
        """Generate DAGNode representing matrix with data either given by a numpy array, which will be sent to SystemDS
        on need, or a path pointing to a matrix.

        :param mat: the numpy array or path to matrix file
        :param args: unnamed parameters
        :param kwargs: named parameters
        """
        if isinstance(mat, str):
            unnamed_params = [f'\'{mat}\'']
            named_params = {}
            self._np_array = None
        else:
            # TODO better alternative than format string?
            unnamed_params = ['\'./tmp/{file_name}\'']
            named_params = {'rows': -1, 'cols': -1}
            self._np_array = mat
        unnamed_params.extend(args)
        named_params.update(kwargs)
        super().__init__(sds_context, 'read', unnamed_params,
                         named_params, is_python_local_data=self._is_numpy())

    def pass_python_data_to_prepared_script(self, jvm: JVMView, var_name: str, prepared_script: JavaObject) -> None:
        assert self.is_python_local_data, 'Can only pass data to prepared script if it is python local!'
        if self._is_numpy():
            prepared_script.setMatrix(var_name, numpy_to_matrix_block(
                jvm, self._np_array), True)  # True for reuse

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

    def _is_numpy(self) -> bool:
        return self._np_array is not None

    def rev(self) -> OperationNode:
        """ Reverses the rows in a matrix

        :return: the OperationNode representing this operation
        """

        self._is_numpy()
        return OperationNode(self.sds_context, 'rev', [self])

    def order(self, by: int = 1, decreasing: bool = False,
              index_return: bool = False) -> OperationNode:
        """ Sort by a column of the matrix X in increasing/decreasing order and returns either the index or data

        :param by: sort matrix by this column number
        :param decreasing: If true the matrix will be sorted in decreasing order
        :param index_return: If true, the index numbers will be returned
        :return: the OperationNode representing this operation
        """

        self._is_numpy()

        cols = self._np_array.shape[1]
        if by > cols:
            raise IndexError("Index {i} is out of bounds for axis 1 with size {c}".format(i=by, c=cols))

        named_input_nodes = {'target': self, 'by': by, 'decreasing': str(decreasing).upper(),
                             'index.return': str(index_return).upper()}

        return OperationNode(self.sds_context, 'order', [], named_input_nodes=named_input_nodes)

    def t(self) -> OperationNode:
        """ Transposes the input matrix

        :return: the OperationNode representing this operation
        """

        self._is_numpy()
        return OperationNode(self.sds_context, 't', [self])

    def cholesky(self, safe: bool = False) -> OperationNode:
        """ Computes the Cholesky decomposition of a symmetric, positive definite matrix

        :param safe: default value is False, if flag is True additional checks to ensure
        that the matrix is symmetric positive definite are applied, if False, checks will be skipped
        :return: the OperationNode representing this operation
        """

        self._is_numpy()

        # check square dimension
        if self._np_array.shape[0] != self._np_array.shape[1]:
            raise ValueError("Last 2 dimensions of the array must be square")

        if safe:
            # check if mat is positive definite
            if not np.all(np.linalg.eigvals(self._np_array) > 0):
                raise ValueError("Matrix is not positive definite")

            # check if mat is symmetric
            if not np.allclose(self._np_array, self._np_array.transpose()):
                raise ValueError("Matrix is not symmetric")

        return OperationNode(self.sds_context, 'cholesky', [self])
