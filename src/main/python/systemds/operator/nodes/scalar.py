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

__all__ = ["Scalar"]

import os
from typing import (TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple,
                    Union)

import numpy as np
from py4j.java_gateway import JavaObject, JVMView
from systemds.operator import OperationNode
from systemds.script_building.dag import OutputType
from systemds.utils.consts import (BINARY_OPERATIONS, VALID_ARITHMETIC_TYPES,
                                   VALID_INPUT_TYPES)
from systemds.utils.converters import numpy_to_matrix_block


class Scalar(OperationNode):
    __assign: bool

    def __init__(self, sds_context: 'SystemDSContext', operation: str,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None,
                 named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
                 output_type: OutputType = OutputType.DOUBLE,
                 assign: bool = False) -> 'Scalar':
        self.__assign = assign
        super().__init__(sds_context, operation, unnamed_input_nodes=unnamed_input_nodes,
                         named_input_nodes=named_input_nodes, output_type=output_type)

    def pass_python_data_to_prepared_script(self, sds, var_name: str, prepared_script: JavaObject) -> None:
        raise RuntimeError(
            'Scalar Operation Nodes, should not have python data input')

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        if self.__assign:
            return f'{var_name}={self.operation};'
        else:
            return super().code_line(var_name, unnamed_input_vars, named_input_vars)

    def compute(self, verbose: bool = False, lineage: bool = False) -> Union[np.array]:
        return super().compute(verbose, lineage)

    def _parse_output_result_variables(self, result_variables):
        if self.output_type == OutputType.DOUBLE:
            return result_variables.getDouble(self._script.out_var_name[0])
        elif self.output_type == OutputType.STRING:
            return result_variables.getString(self._script.out_var_name[0])
        else:
            raise NotImplemented(
                "Not currently support scalar type: " + self.output_type)

    def __add__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.sds_context, '+', [self, other])

    # Left hand side
    def __radd__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.sds_context, '+', [other, self])

    def __sub__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.sds_context, '-', [self, other])

    # Left hand side
    def __rsub__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.sds_context, '-', [other, self])

    def __mul__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.sds_context, '*', [self, other])

    def __rmul__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.sds_context, '*', [other, self])

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.sds_context, '/', [self, other])

    def __rtruediv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.sds_context, '/', [other, self])

    def __floordiv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.sds_context, '//', [self, other])

    def __rfloordiv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.sds_context, '//', [other, self])

    def __lt__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '<', [self, other])

    def __rlt__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '<', [other, self])

    def __le__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '<=', [self, other])

    def __rle__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '<=', [other, self])

    def __gt__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '>', [self, other])

    def __rgt__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '>', [other, self])

    def __ge__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '>=', [self, other])

    def __rge__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '>=', [other, self])

    def __eq__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '==', [self, other])

    def __req__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '==', [other, self])

    def __ne__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '!=', [self, other])

    def __rne__(self, other) -> 'Scalar':
        return Scalar(self.sds_context, '!=', [other, self])

    def __matmul__(self, other: 'Scalar') -> 'Scalar':
        return Scalar(self.sds_context, '%*%', [self, other])

    def sum(self) -> 'Scalar':
        return Scalar(self.sds_context, 'sum', [self], output_type=OutputType.DOUBLE)

    def mean(self) -> 'Scalar':
        return Scalar(self.sds_context, 'mean', [self], output_type=OutputType.DOUBLE)

    def var(self, axis: int = None) -> 'Scalar':
        return Scalar(self.sds_context, 'var', [self], output_type=OutputType.DOUBLE)

    def abs(self) -> 'Scalar':
        """Calculate absolute.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, 'abs', [self])

    def sin(self) -> 'Scalar':
        """Calculate sin.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, 'sin', [self])

    def cos(self) -> 'Scalar':
        """Calculate cos.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, 'cos', [self])

    def tan(self) -> 'Scalar':
        """Calculate tan.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, 'tan', [self])

    def asin(self) -> 'Scalar':
        """Calculate arcsin.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, 'asin', [self])

    def acos(self) -> 'Scalar':
        """Calculate arccos.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, 'acos', [self])

    def atan(self) -> 'Scalar':
        """Calculate arctan.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, 'atan', [self])

    def sinh(self) -> 'Scalar':
        """Calculate sin.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, 'sinh', [self])

    def cosh(self) -> 'Scalar':
        """Calculate cos.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, 'cosh', [self])

    def tanh(self) -> 'Scalar':
        """Calculate tan.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, 'tanh', [self])

    def to_string(self, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'Scalar':
        """ Converts the input to a string representation.
        :return: `Scalar` containing the string.
        """
        return Scalar(self.sds_context, 'toString', [self], named_input_nodes=kwargs, output_type=OutputType.STRING)

    def __str__(self):
        return "ScalarNode"
