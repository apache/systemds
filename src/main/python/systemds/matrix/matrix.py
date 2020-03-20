# ------------------------------------------------------------------------------
#  Copyright 2020 Graz University of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ------------------------------------------------------------------------------

__all__ = ['Matrix', 'federated', 'full', 'seq']

import os
from typing import Union, Optional, Iterable, Dict, Tuple, Sequence

import numpy as np
from py4j.java_gateway import JVMView, JavaObject

from ..utils.converters import numpy_to_matrix_block
from ..script_building.dag import VALID_INPUT_TYPES
from .operation_node import OperationNode


# TODO maybe instead of having a new class we could have a function `matrix` instead, adding behaviour to
#  `OperationNode` would be necessary
class Matrix(OperationNode):
    np_array: Optional[np.array]

    def __init__(self, mat: Union[np.array, os.PathLike], *args: Sequence[VALID_INPUT_TYPES],
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
            self.np_array = None
        else:
            unnamed_params = ['\'./tmp/{file_name}\'']  # TODO better alternative than format string?
            named_params = {'rows': -1, 'cols': -1}
            self.np_array = mat
        unnamed_params.extend(args)
        named_params.update(kwargs)
        super().__init__('read', unnamed_params, named_params, is_python_local_data=self._is_numpy())

    def pass_python_data_to_prepared_script(self, jvm: JVMView, var_name: str, prepared_script: JavaObject) -> None:
        assert self.is_python_local_data, 'Can only pass data to prepared script if it is python local!'
        if self._is_numpy():
            prepared_script.setMatrix(var_name, numpy_to_matrix_block(jvm, self.np_array), True)  # True for reuse

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        code_line = super().code_line(var_name, unnamed_input_vars, named_input_vars)
        if self._is_numpy():
            code_line = code_line.format(file_name=var_name)
        return code_line

    def compute(self, verbose: bool = False) -> Union[np.array]:
        if self._is_numpy():
            if verbose:
                print('[Numpy Array - No Compilation necessary]')
            return self.np_array
        else:
            return super().compute(verbose)

    def _is_numpy(self) -> bool:
        return self.np_array is not None


def federated(addresses: Iterable[str], ranges: Iterable[Tuple[Iterable[int], Iterable[int]]], *args,
              **kwargs) -> OperationNode:
    """Create federated matrix object.

    :param addresses: addresses of the federated workers
    :param ranges: for each federated worker a pair of begin and end index of their held matrix
    :param args: unnamed params
    :param kwargs: named params
    :return: the OperationNode representing this operation
    """
    addresses_str = 'list(' + ','.join(map(lambda s: f'"{s}"', addresses)) + ')'
    ranges_str = 'list('
    for begin, end in ranges:
        ranges_str += f'list({",".join(map(str, begin))}), list({",".join(map(str, end))})'
    ranges_str += ')'
    named_params = {'addresses': addresses_str, 'ranges': ranges_str}
    named_params.update(kwargs)
    return OperationNode('federated', args, named_params)


def full(shape: Tuple[int, int], value: Union[float, int]) -> OperationNode:
    """Generates a matrix completely filled with a value

    :param shape: shape (rows and cols) of the matrix TODO tensor
    :param value: the value to fill all cells with
    :return: the OperationNode representing this operation
    """
    unnamed_input_nodes = [value]
    named_input_nodes = {'rows': shape[0], 'cols': shape[1]}
    return OperationNode('matrix', unnamed_input_nodes, named_input_nodes)


def seq(start: Union[float, int], stop: Union[float, int] = None, step: Union[float, int] = 1) -> OperationNode:
    """Create a single column vector with values from `start` to `stop` and an increment of `step`.
    If no stop is defined and only one parameter is given, then start will be 0 and the parameter will be interpreted as
    stop.

    :param start: the starting value
    :param stop: the maximum value
    :param step: the step size
    :return: the OperationNode representing this operation
    """
    if stop is None:
        stop = start
        start = 0
    unnamed_input_nodes = [start, stop, step]
    return OperationNode('seq', unnamed_input_nodes)
