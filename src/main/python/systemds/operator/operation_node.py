#-------------------------------------------------------------
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
#-------------------------------------------------------------

from typing import Union, Optional, Iterable, Dict, Sequence, Tuple, TYPE_CHECKING

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
    _result_var: Optional[Union[float, np.array]]
    _lineage_trace: Optional[str]
    _script: Optional[DMLScript]

    def __init__(self, sds_context: 'SystemDSContext', operation: str,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None,
                 named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
                 output_type: OutputType = OutputType.MATRIX, is_python_local_data: bool = False):
        """
        Create general `OperationNode`

        :param sds_context: The SystemDS context for performing the operations
        :param operation: The name of the DML function to execute
        :param unnamed_input_nodes: inputs identified by their position, not name
        :param named_input_nodes: inputs with their respective parameter name
        :param output_type: type of the output in DML (double, matrix etc.)
        :param is_python_local_data: if the data is local in python e.g. numpy arrays
        """
        self.sds_context = sds_context
        if unnamed_input_nodes is None:
            unnamed_input_nodes = []
        if named_input_nodes is None:
            named_input_nodes = {}
        self.operation = operation
        self._unnamed_input_nodes = unnamed_input_nodes
        self._named_input_nodes = named_input_nodes
        self.output_type = output_type
        self._is_python_local_data = is_python_local_data
        self._result_var = None
        self._lineage_trace = None
        self._script = None

    def compute(self, verbose: bool = False, lineage: bool = False) -> \
            Union[float, np.array, Tuple[Union[float, np.array], str]]:
        if self._result_var is None or self._lineage_trace is None:
            self._script = DMLScript(self.sds_context)
            self._script.build_code(self)
            if lineage:
                result_variables, self._lineage_trace = self._script.execute(lineage)
            else:
                result_variables = self._script.execute(lineage)
            if self.output_type == OutputType.DOUBLE:
                self._result_var = result_variables.getDouble(self._script.out_var_name)
            elif self.output_type == OutputType.MATRIX:
                self._result_var = matrix_block_to_numpy(self.sds_context.java_gateway.jvm,
                                                         result_variables.getMatrixBlock(self._script.out_var_name))
        if verbose:
            print(self._script.dml_script)
            # TODO further info

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
        return OperationNode(self.sds_context, '+', [self, other])

    def __sub__(self, other: VALID_ARITHMETIC_TYPES):
        return OperationNode(self.sds_context, '-', [self, other])

    def __mul__(self, other: VALID_ARITHMETIC_TYPES):
        return OperationNode(self.sds_context, '*', [self, other])

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES):
        return OperationNode(self.sds_context, '/', [self, other])

    def __floordiv__(self, other: VALID_ARITHMETIC_TYPES):
        return OperationNode(self.sds_context, '//', [self, other])

    def __lt__(self, other) -> 'OperationNode':
        return OperationNode(self.sds_context, '<', [self, other])

    def __le__(self, other):
        return OperationNode(self.sds_context, '<=', [self, other])

    def __gt__(self, other):
        return OperationNode(self.sds_context, '>', [self, other])

    def __ge__(self, other):
        return OperationNode(self.sds_context, '>=', [self, other])

    def __eq__(self, other):
        return OperationNode(self.sds_context, '==', [self, other])

    def __ne__(self, other):
        return OperationNode(self.sds_context, '!=', [self, other])

    def __matmul__(self, other: VALID_ARITHMETIC_TYPES):
        return OperationNode(self.sds_context, '%*%', [self, other])

    def sum(self, axis: int = None) -> 'OperationNode':
        """Calculate sum of matrix.

        :param axis: can be 0 or 1 to do either row or column sums
        :return: `OperationNode` representing operation
        """
        self._check_matrix_op()
        if axis == 0:
            return OperationNode(self.sds_context, 'colSums', [self])
        elif axis == 1:
            return OperationNode(self.sds_context, 'rowSums', [self])
        elif axis is None:
            return OperationNode(self.sds_context, 'sum', [self], output_type=OutputType.DOUBLE)
        raise ValueError(f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def mean(self, axis: int = None) -> 'OperationNode':
        """Calculate mean of matrix.

        :param axis: can be 0 or 1 to do either row or column means
        :return: `OperationNode` representing operation
        """
        self._check_matrix_op()
        if axis == 0:
            return OperationNode(self.sds_context, 'colMeans', [self])
        elif axis == 1:
            return OperationNode(self.sds_context, 'rowMeans', [self])
        elif axis is None:
            return OperationNode(self.sds_context, 'mean', [self], output_type=OutputType.DOUBLE)
        raise ValueError(f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def var(self, axis: int = None) -> 'OperationNode':
        """Calculate variance of matrix.

        :param axis: can be 0 or 1 to do either row or column vars
        :return: `OperationNode` representing operation
        """
        self._check_matrix_op()
        if axis == 0:
            return OperationNode(self.sds_context, 'colVars', [self])
        elif axis == 1:
            return OperationNode(self.sds_context, 'rowVars', [self])
        elif axis is None:
            return OperationNode(self.sds_context, 'var', [self], output_type=OutputType.DOUBLE)
        raise ValueError(f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}")

    def abs(self) -> 'OperationNode':
        """Calculate absolute.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'abs', [self])

    def sin(self) -> 'OperationNode':
        """Calculate sin.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'sin', [self])

    def cos(self) -> 'OperationNode':
        """Calculate cos.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'cos', [self])

    def tan(self) -> 'OperationNode':
        """Calculate tan.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'tan', [self])

    def asin(self) -> 'OperationNode':
        """Calculate arcsin.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'asin', [self])

    def acos(self) -> 'OperationNode':
        """Calculate arccos.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'acos', [self])

    def atan(self) -> 'OperationNode':
        """Calculate arctan.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'atan', [self])

    def sinh(self) -> 'OperationNode':
        """Calculate sin.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'sinh', [self])

    def cosh(self) -> 'OperationNode':
        """Calculate cos.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'cosh', [self])

    def tanh(self) -> 'OperationNode':
        """Calculate tan.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'tanh', [self])
    '''
    def rev(self) -> 'OperationNode':
        """Calculate tan.

        :return: `OperationNode` representing operation
        """
        return OperationNode(self.sds_context, 'rev', [self])
    '''

    def moment(self, moment, weights: DAGNode = None) -> 'OperationNode':
        # TODO write tests
        self._check_matrix_op()
        unnamed_inputs = [self]
        if weights is not None:
            unnamed_inputs.append(weights)
        unnamed_inputs.append(moment)
        return OperationNode(self.sds_context, 'moment', unnamed_inputs, output_type=OutputType.DOUBLE)

    def lm(self, y: DAGNode, **kwargs) -> 'OperationNode':
        self._check_matrix_op()

        if self._np_array.size == 0:
            raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                             .format(s=self._np_array.shape))

        if y._np_array.size == 0:
            raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                             .format(s=y._np_array.shape))

        params_dict = {'X': self, 'y': y}
        params_dict.update(kwargs)

        return OperationNode(self.sds_context, 'lm', named_input_nodes=params_dict)