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
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from py4j.java_gateway import JavaObject, JVMView
from systemds.operator.operation_node import OperationNode
from systemds.utils.consts import (
    BINARY_OPERATIONS,
    VALID_ARITHMETIC_TYPES,
    VALID_INPUT_TYPES,
)
from systemds.utils.converters import numpy_to_matrix_block


class Scalar(OperationNode):
    __assign: bool

    def __init__(
        self,
        sds_context,
        operation: str,
        unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None,
        named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
        assign: bool = False,
    ) -> "Scalar":
        self.__assign = assign
        super().__init__(
            sds_context,
            operation,
            unnamed_input_nodes=unnamed_input_nodes,
            named_input_nodes=named_input_nodes,
            is_datatype_none=False,
        )

    def pass_python_data_to_prepared_script(
        self, sds, var_name: str, prepared_script: JavaObject
    ) -> None:
        raise RuntimeError("Scalar Operation Nodes, should not have python data input")

    def code_line(
        self,
        var_name: str,
        unnamed_input_vars: Sequence[str],
        named_input_vars: Dict[str, str],
    ) -> str:
        if self.__assign:
            return f"{var_name}={self.operation};"
        else:
            return super().code_line(var_name, unnamed_input_vars, named_input_vars)

    def compute(self, verbose: bool = False, lineage: bool = False):
        return super().compute(verbose, lineage)

    def _parse_output_result_variables(self, result_variables):
        scalar_object = result_variables.getScalarObject(self._script.out_var_name[0])
        value_type = scalar_object.getValueType().toString()
        if value_type in ["FP64", "FP32"]:
            return scalar_object.getDoubleValue()
        elif value_type == "STRING":
            return scalar_object.getStringValue()
        elif value_type in ["INT64", "INT32"]:
            return scalar_object.getLongValue()
        elif value_type == "BOOLEAN":
            return scalar_object.getBooleanValue()
        else:
            raise NotImplementedError(
                "Not currently support scalar type: " + value_type
            )

    def __add__(self, other: VALID_ARITHMETIC_TYPES) -> "Scalar":
        return Scalar(self.sds_context, "+", [self, other])

    # Left hand side
    def __radd__(self, other: VALID_ARITHMETIC_TYPES) -> "Scalar":
        return Scalar(self.sds_context, "+", [other, self])

    def __sub__(self, other: VALID_ARITHMETIC_TYPES) -> "Scalar":
        return Scalar(self.sds_context, "-", [self, other])

    # Left hand side
    def __rsub__(self, other: VALID_ARITHMETIC_TYPES) -> "Scalar":
        return Scalar(self.sds_context, "-", [other, self])

    def __mul__(self, other: VALID_ARITHMETIC_TYPES) -> "Scalar":
        return Scalar(self.sds_context, "*", [self, other])

    def __rmul__(self, other: VALID_ARITHMETIC_TYPES) -> "Scalar":
        return Scalar(self.sds_context, "*", [other, self])

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES) -> "Scalar":
        return Scalar(self.sds_context, "/", [self, other])

    def __rtruediv__(self, other: VALID_ARITHMETIC_TYPES) -> "Scalar":
        return Scalar(self.sds_context, "/", [other, self])

    def __floordiv__(self, other: VALID_ARITHMETIC_TYPES) -> "Scalar":
        return Scalar(self.sds_context, "//", [self, other])

    def __rfloordiv__(self, other: VALID_ARITHMETIC_TYPES) -> "Scalar":
        return Scalar(self.sds_context, "//", [other, self])

    def __lt__(self, other) -> "Scalar":
        return Scalar(self.sds_context, "<", [self, other])

    def __rlt__(self, other) -> "Scalar":
        return Scalar(self.sds_context, "<", [other, self])

    def __le__(self, other) -> "Scalar":
        return Scalar(self.sds_context, "<=", [self, other])

    def __rle__(self, other) -> "Scalar":
        return Scalar(self.sds_context, "<=", [other, self])

    def __gt__(self, other) -> "Scalar":
        return Scalar(self.sds_context, ">", [self, other])

    def __rgt__(self, other) -> "Scalar":
        return Scalar(self.sds_context, ">", [other, self])

    def __ge__(self, other) -> "Scalar":
        return Scalar(self.sds_context, ">=", [self, other])

    def __rge__(self, other) -> "Scalar":
        return Scalar(self.sds_context, ">=", [other, self])

    def __eq__(self, other) -> "Scalar":
        return Scalar(self.sds_context, "==", [self, other])

    def __req__(self, other) -> "Scalar":
        return Scalar(self.sds_context, "==", [other, self])

    def __ne__(self, other) -> "Scalar":
        return Scalar(self.sds_context, "!=", [self, other])

    def __rne__(self, other) -> "Scalar":
        return Scalar(self.sds_context, "!=", [other, self])

    def __matmul__(self, other: "Scalar") -> "Scalar":
        return Scalar(self.sds_context, "%*%", [self, other])

    def sum(self) -> "Scalar":
        return Scalar(self.sds_context, "sum", [self])

    def mean(self) -> "Scalar":
        return Scalar(self.sds_context, "mean", [self])

    def var(self, axis: int = None) -> "Scalar":
        return Scalar(self.sds_context, "var", [self])

    def abs(self) -> "Scalar":
        """Calculate absolute.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "abs", [self])

    def sqrt(self) -> "Scalar":
        """Calculate square root.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "sqrt", [self])

    def floor(self) -> "Scalar":
        """Return the floor of the input, element-wise.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "floor", [self])

    def ceil(self) -> "Scalar":
        """Return the ceiling of the input, element-wise.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "ceil", [self])

    def log(self) -> "Scalar":
        """Calculate logarithm.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "log", [self])

    def sin(self) -> "Scalar":
        """Calculate sin.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "sin", [self])

    def exp(self) -> "Scalar":
        """Calculate exponential.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "exp", [self])

    def sign(self) -> "Scalar":
        """Returns a the signs of the input,
        where 1 represents positive, 0 represents zero, and -1 represents negative.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "sign", [self])

    def cos(self) -> "Scalar":
        """Calculate cos.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "cos", [self])

    def tan(self) -> "Scalar":
        """Calculate tan.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "tan", [self])

    def asin(self) -> "Scalar":
        """Calculate arcsin.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "asin", [self])

    def acos(self) -> "Scalar":
        """Calculate arccos.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "acos", [self])

    def atan(self) -> "Scalar":
        """Calculate arctan.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "atan", [self])

    def sinh(self) -> "Scalar":
        """Calculate sin.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "sinh", [self])

    def cosh(self) -> "Scalar":
        """Calculate cos.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "cosh", [self])

    def tanh(self) -> "Scalar":
        """Calculate tan.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "tanh", [self])

    def to_string(self, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> "Scalar":
        """Converts the input to a string representation.
        :return: `Scalar` containing the string.
        """
        return Scalar(self.sds_context, "toString", [self], named_input_nodes=kwargs)

    def isNA(self) -> "Scalar":
        """Computes a boolean indicator matrix of the same shape as the input, indicating where NA (not available)
        values are located. Currently, NA is only capturing NaN values.

        :return: the OperationNode representing this operation
        """
        return Scalar(self.sds_context, "isNA", [self])

    def isNaN(self) -> "Scalar":
        """Computes a boolean indicator matrix of the same shape as the input, indicating where NaN (not a number)
        values are located.

        :return: the OperationNode representing this operation
        """
        return Scalar(self.sds_context, "isNaN", [self])

    def isInf(self) -> "Scalar":
        """Computes a boolean indicator matrix of the same shape as the input, indicating where Inf (positive or
        negative infinity) values are located.
        :return: the OperationNode representing this operation
        """
        return Scalar(self.sds_context, "isInf", [self])

    def __str__(self):
        return "ScalarNode"
