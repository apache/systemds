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

import numpy as np
from py4j.java_gateway import JVMView, JavaObject

from ..utils.helpers import get_gateway, create_params_string
from ..utils.converters import matrix_block_to_numpy
from ..script_building.script import DMLScript
from ..script_building.dag import OutputType, DAGNode, VALID_INPUT_TYPES
from typing import Union, Optional, Iterable, Dict, Sequence

BINARY_OPERATIONS = ['+', '-', '/', '//', '*', '<', '<=', '>', '>=', '==', '!=']
# TODO add numpy array
VALID_ARITHMETIC_TYPES = Union[DAGNode, int, float]

__all__ = ["OperationNode"]


class OperationNode(DAGNode):
    result_var: Optional[Union[float, np.array]]
    script: Optional[DMLScript]

    def __init__(self, operation: str, unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None,
                 named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
                 output_type: OutputType = OutputType.MATRIX, is_python_local_data: bool = False):
        """
        Create general `OperationNode`

        :param operation: The name of the DML function to execute
        :param unnamed_input_nodes: inputs identified by their position, not name
        :param named_input_nodes: inputs with their respective parameter name
        :param output_type: type of the output in DML (double, matrix etc.)
        :param is_python_local_data: if the data is local in python e.g. numpy arrays
        """
        if unnamed_input_nodes is None:
            unnamed_input_nodes = []
        if named_input_nodes is None:
            named_input_nodes = {}
        self.operation = operation
        self.unnamed_input_nodes = unnamed_input_nodes
        self.named_input_nodes = named_input_nodes
        self.output_type = output_type
        self.is_python_local_data = is_python_local_data
        self.result_var = None
        self.script = None

    def compute(self, verbose: bool = False) -> Union[float, np.array]:
        if self.result_var is None:
            self.script = DMLScript()
            self.script.build_code(self)
            result_variables = self.script.execute()
            if self.output_type == OutputType.DOUBLE:
                self.result_var = result_variables.getDouble(self.script.out_var_name)
            elif self.output_type == OutputType.MATRIX:
                self.result_var = matrix_block_to_numpy(get_gateway().jvm,
                                                        result_variables.getMatrixBlock(self.script.out_var_name))
        if verbose:
            print(self.script.dml_script)
            # TODO further info
        return self.result_var

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        if self.operation in BINARY_OPERATIONS:
            assert len(named_input_vars) == 0, 'Named parameters can not be used with binary operations'
            assert len(unnamed_input_vars) == 2, 'Binary Operations need exactly two input variables'
            return f'{var_name}={unnamed_input_vars[0]}{self.operation}{unnamed_input_vars[1]}'
        else:
            inputs_comma_sep = create_params_string(unnamed_input_vars, named_input_vars)
            return f'{var_name}={self.operation}({inputs_comma_sep});'

    def pass_python_data_to_prepared_script(self, jvm: JVMView, var_name: str, prepared_script: JavaObject) -> None:
        raise NotImplementedError('Operation node has no python local data. Missing implementation in derived class?')

    def _check_matrix_op(self):
        """Perform checks to assure operation is allowed to be performed on data type of this `OperationNode`

        :raise: AssertionError
        """
        assert self.output_type == OutputType.MATRIX, f'{self.operation} only supported for matrices'

    def __add__(self, other: VALID_ARITHMETIC_TYPES):
        return OperationNode('+', [self, other])

    def __sub__(self, other: VALID_ARITHMETIC_TYPES):
        return OperationNode('-', [self, other])

    def __mul__(self, other: VALID_ARITHMETIC_TYPES):
        return OperationNode('*', [self, other])

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES):
        return OperationNode('/', [self, other])

    def __floordiv__(self, other: VALID_ARITHMETIC_TYPES):
        return OperationNode('//', [self, other])

    def __lt__(self, other) -> 'OperationNode':
        return OperationNode('<', [self, other])

    def __le__(self, other):
        return OperationNode('<=', [self, other])

    def __gt__(self, other):
        return OperationNode('>', [self, other])

    def __ge__(self, other):
        return OperationNode('>=', [self, other])

    def __eq__(self, other):
        return OperationNode('==', [self, other])

    def __ne__(self, other):
        return OperationNode('!=', [self, other])

    def l2svm(self, labels: DAGNode, **kwargs) -> 'OperationNode':
        """Perform l2svm on matrix with labels given.

        :return: `OperationNode` representing operation
        """
        self._check_matrix_op()
        params_dict = {'X': self, 'Y': labels}
        params_dict.update(kwargs)
        return OperationNode('l2svm', named_input_nodes=params_dict)

    def sum(self, axis: int = None) -> 'OperationNode':
        """Calculate sum of matrix.

        :param axis: can be 0 or 1 to do either row or column sums
        :return: `OperationNode` representing operation
        """
        self._check_matrix_op()
        if axis == 0:
            return OperationNode('colSums', [self])
        elif axis == 1:
            return OperationNode('rowSums', [self])
        elif axis is None:
            return OperationNode('sum', [self], output_type=OutputType.DOUBLE)
        raise ValueError(f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def mean(self, axis: int = None) -> 'OperationNode':
        """Calculate mean of matrix.

        :param axis: can be 0 or 1 to do either row or column means
        :return: `OperationNode` representing operation
        """
        self._check_matrix_op()
        if axis == 0:
            return OperationNode('colMeans', [self])
        elif axis == 1:
            return OperationNode('rowMeans', [self])
        elif axis is None:
            return OperationNode('mean', [self], output_type=OutputType.DOUBLE)
        raise ValueError(f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def var(self, axis: int = None) -> 'OperationNode':
        """Calculate variance of matrix.

        :param axis: can be 0 or 1 to do either row or column vars
        :return: `OperationNode` representing operation
        """
        self._check_matrix_op()
        if axis == 0:
            return OperationNode('colVars', [self])
        elif axis == 1:
            return OperationNode('rowVars', [self])
        elif axis is None:
            return OperationNode('var', [self], output_type=OutputType.DOUBLE)
        raise ValueError(f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def abs(self) -> 'OperationNode':
        """Calculate absolute.

        :return: `OperationNode` representing operation
        """
        return OperationNode('abs', [self])

    def moment(self, moment, weights: DAGNode = None) -> 'OperationNode':
        # TODO write tests
        self._check_matrix_op()
        unnamed_inputs = [self]
        if weights is not None:
            unnamed_inputs.append(weights)
        unnamed_inputs.append(moment)
        return OperationNode('moment', unnamed_inputs, output_type=OutputType.DOUBLE)
