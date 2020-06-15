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

__all__ = ['full', 'seq', 'rand', 'rev', 'order', 't', 'cholesky']

'''
Contains a number of different data generators
'''

from typing import Union, Tuple

from systemds.operator import OperationNode
from systemds.context import SystemDSContext
from systemds.matrix import Matrix
import numpy as np

def full(sds_context: SystemDSContext, shape: Tuple[int, int], value: Union[float, int]) -> OperationNode:
    """Generates a matrix completely filled with a value

    :param sds_context: SystemDS context
    :param shape: shape (rows and cols) of the matrix TODO tensor
    :param value: the value to fill all cells with
    :return: the OperationNode representing this operation
    """
    unnamed_input_nodes = [value]
    named_input_nodes = {'rows': shape[0], 'cols': shape[1]}
    return OperationNode(sds_context, 'matrix', unnamed_input_nodes, named_input_nodes)


def seq(sds_context: SystemDSContext, start: Union[float, int], stop: Union[float, int] = None,
        step: Union[float, int] = 1) -> OperationNode:
    """Create a single column vector with values from `start` to `stop` and an increment of `step`.
    If no stop is defined and only one parameter is given, then start will be 0 and the parameter will be interpreted as
    stop.

    :param sds_context: SystemDS context
    :param start: the starting value
    :param stop: the maximum value
    :param step: the step size
    :return: the OperationNode representing this operation
    """
    if stop is None:
        stop = start
        start = 0
    unnamed_input_nodes = [start, stop, step]
    return OperationNode(sds_context, 'seq', unnamed_input_nodes)


def rand(sds_context: SystemDSContext, rows: int, cols: int,
         min: Union[float, int] = None, max: Union[float, int] = None, pdf: str = "uniform",
         sparsity: Union[float, int] = None, seed: Union[float, int] = None,
         lambd: Union[float, int] = 1) -> OperationNode:
    """Generates a matrix filled with random values

    :param sds_context: SystemDS context
    :param rows: number of rows
    :param cols: number of cols
    :param min: min value for cells
    :param max: max value for cells
    :param pdf: "uniform"/"normal"/"poison" distribution
    :param sparsity: fraction of non-zero cells
    :param seed: random seed
    :param lambd: lamda value for "poison" distribution
    :return:
    """
    available_pdfs = ["uniform", "normal", "poisson"]
    if rows < 0:
        raise ValueError("In rand statement, can only assign rows a long (integer) value >= 0 "
                         "-- attempted to assign value: {r}".format(r=rows))
    if cols < 0:
        raise ValueError("In rand statement, can only assign cols a long (integer) value >= 0 "
                         "-- attempted to assign value: {c}".format(c=cols))
    if pdf not in available_pdfs:
        raise ValueError("The pdf passed is invalid! given: {g}, expected: {e}".format(
            g=pdf, e=available_pdfs))

    pdf = '\"' + pdf + '\"'
    named_input_nodes = {
        'rows': rows, 'cols': cols, 'pdf': pdf, 'lambda': lambd}
    if min is not None:
        named_input_nodes['min'] = min
    if max is not None:
        named_input_nodes['max'] = max
    if sparsity is not None:
        named_input_nodes['sparsity'] = sparsity
    if seed is not None:
        named_input_nodes['seed'] = seed

    return OperationNode(sds_context, 'rand', [], named_input_nodes=named_input_nodes)


def rev(sds_context: 'SystemDSContext', X: Matrix) -> 'OperationNode':
    """ Reverses the rows in a matrix

    :param sds_context: SystemDS context
    :param X: Input matrix
    :return: reversed matrix
    """
    X._check_matrix_op()
    return OperationNode(sds_context, 'rev', [X])


def order(sds_context: 'SystemDSContext', X: Matrix, by: int = 1, decreasing: bool = False, index_return: bool = False) -> 'OperationNode':
    """ Sort by a column of the matrix X in increasing/decreasing order and returns either the index or data

    :param sds_context: SystemDS context
    :param X: Input matrix
    :param by: Column number
    :param decreasing: If true the matrix will be sorted in decreasing order
    :param index_return: If true, theindex numbers will be returned
    :return: sorted matrix
    """
    X._check_matrix_op()

    cols = X._np_array.shape[1]
    if by > cols:
        raise IndexError("Index {i} is out of bounds for axis 1 with size {c}".format(i=by, c=cols))

    named_input_nodes = {'target': X, 'by': by, 'decreasing': str(decreasing).upper(), 'index.return': str(index_return).upper()}

    return OperationNode(sds_context, 'order', [], named_input_nodes=named_input_nodes)


def t(sds_context: 'SystemDSContext', X: Matrix) -> 'OperationNode':
    """ Transposes the input matrix

    :param sds_context: SystemDS context
    :param X: Input matrix
    :return: transposed matrix
    """
    X._check_matrix_op()
    return OperationNode(sds_context, 't', [X])


def cholesky(sds_context: 'SystemDSContext', X: Matrix) -> 'OperationNode':
    """ Computes the Cholesky decomposition of a symmetric, positive definite matrix

    :param sds_context: SystemDS context
    :param X: Input matrix
    :return: Cholesky decomposition
    """
    X._check_matrix_op()

    # check square dimension
    if X._np_array.shape[0] != X._np_array.shape[1]:
        raise ValueError("Last 2 dimensions of the array must be square")

    # check if mat is positive definite
    if not np.all(np.linalg.eigvals(X._np_array)>0):
        raise ValueError("Matrix is not positive definite")

    # check if mat is symmetric
    if not np.allclose(X._np_array, X._np_array.transpose()):
        raise ValueError("Matrix is not symmetric")

    return OperationNode(sds_context, 'cholesky', [X])