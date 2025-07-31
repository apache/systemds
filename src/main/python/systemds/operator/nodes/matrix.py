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

from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Union

import numpy as np
from py4j.java_gateway import JavaObject
from systemds.operator.operation_node import OperationNode
from systemds.operator.nodes.multi_return import MultiReturn
from systemds.operator.nodes.scalar import Scalar
from systemds.utils.consts import (
    BINARY_OPERATIONS,
    VALID_ARITHMETIC_TYPES,
    VALID_INPUT_TYPES,
)
from systemds.utils.converters import matrix_block_to_numpy, numpy_to_matrix_block
from systemds.utils.helpers import (
    check_is_empty_slice,
    check_no_less_than_zero,
    get_slice_string,
)


def to_matrix(self):
    return Matrix(self.sds_context, "as.matrix", [self])


OperationNode.to_matrix = to_matrix


class Matrix(OperationNode):
    _np_array: np.array

    def __init__(
        self,
        sds_context,
        operation: str,
        unnamed_input_nodes: Union[str, Iterable[VALID_INPUT_TYPES]] = None,
        named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
        local_data: np.array = None,
        brackets: bool = False,
    ) -> "Matrix":

        is_python_local_data = False
        if local_data is not None:
            self._np_array = local_data
            is_python_local_data = True
        else:
            self._np_array = None

        super().__init__(
            sds_context,
            operation,
            unnamed_input_nodes,
            named_input_nodes,
            is_python_local_data,
            brackets,
            is_datatype_none=False,
        )

    def pass_python_data_to_prepared_script(
        self, sds, var_name: str, prepared_script: JavaObject
    ) -> None:
        assert (
            self.is_python_local_data
        ), "Can only pass data to prepared script if it is python local!"
        if self._is_numpy():
            prepared_script.setMatrix(
                var_name, numpy_to_matrix_block(sds, self._np_array), True
            )  # True for reuse

    def code_line(
        self,
        var_name: str,
        unnamed_input_vars: Sequence[str],
        named_input_vars: Dict[str, str],
    ) -> str:
        code_line = super().code_line(var_name, unnamed_input_vars, named_input_vars)
        if self._is_numpy():
            code_line = code_line.format(file_name=var_name)
        return code_line

    def compute(self, verbose: bool = False, lineage: bool = False) -> np.array:
        if self._is_numpy():
            self.sds_context._log.info("Numpy Array - No Compilation necessary")
            return self._np_array
        else:
            return super().compute(verbose, lineage)

    def _parse_output_result_variables(self, result_variables):
        return matrix_block_to_numpy(
            self.sds_context,
            result_variables.getMatrixBlock(self._script.out_var_name[0]),
        )

    def _is_numpy(self) -> bool:
        return self._np_array is not None

    def __add__(self, other: VALID_ARITHMETIC_TYPES) -> "Matrix":
        return Matrix(self.sds_context, "+", [self, other])

    # Left hand side
    def __radd__(self, other: VALID_ARITHMETIC_TYPES) -> "Matrix":
        return Matrix(self.sds_context, "+", [other, self])

    def __sub__(self, other: VALID_ARITHMETIC_TYPES) -> "Matrix":
        return Matrix(self.sds_context, "-", [self, other])

    # Left hand side
    def __rsub__(self, other: VALID_ARITHMETIC_TYPES) -> "Matrix":
        return Matrix(self.sds_context, "-", [other, self])

    def __mul__(self, other: VALID_ARITHMETIC_TYPES) -> "Matrix":
        return Matrix(self.sds_context, "*", [self, other])

    def __rmul__(self, other: VALID_ARITHMETIC_TYPES) -> "Matrix":
        return Matrix(self.sds_context, "*", [other, self])

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES) -> "Matrix":
        return Matrix(self.sds_context, "/", [self, other])

    def __rtruediv__(self, other: VALID_ARITHMETIC_TYPES) -> "Matrix":
        return Matrix(self.sds_context, "/", [other, self])

    def __floordiv__(self, other: VALID_ARITHMETIC_TYPES) -> "Matrix":
        return Matrix(self.sds_context, "//", [self, other])

    def __rfloordiv__(self, other: VALID_ARITHMETIC_TYPES) -> "Matrix":
        return Matrix(self.sds_context, "//", [other, self])

    def __lt__(self, other) -> "Matrix":
        return Matrix(self.sds_context, "<", [self, other])

    def __rlt__(self, other) -> "Matrix":
        return Matrix(self.sds_context, "<", [other, self])

    def __le__(self, other) -> "Matrix":
        return Matrix(self.sds_context, "<=", [self, other])

    def __rle__(self, other) -> "Matrix":
        return Matrix(self.sds_context, "<=", [other, self])

    def __gt__(self, other) -> "Matrix":
        return Matrix(self.sds_context, ">", [self, other])

    def __rgt__(self, other) -> "Matrix":
        return Matrix(self.sds_context, ">", [other, self])

    def __ge__(self, other) -> "Matrix":
        return Matrix(self.sds_context, ">=", [self, other])

    def __rge__(self, other) -> "Matrix":
        return Matrix(self.sds_context, ">=", [other, self])

    def __eq__(self, other) -> "Matrix":
        return Matrix(self.sds_context, "==", [self, other])

    def __req__(self, other) -> "Matrix":
        return Matrix(self.sds_context, "==", [other, self])

    def __ne__(self, other) -> "Matrix":
        return Matrix(self.sds_context, "!=", [self, other])

    def __rne__(self, other) -> "Matrix":
        return Matrix(self.sds_context, "!=", [other, self])

    def __matmul__(self, other: "Matrix") -> "Matrix":
        return Matrix(self.sds_context, "%*%", [self, other])

    def nRow(self) -> "Scalar":
        return Scalar(self.sds_context, "nrow", [self])

    def nCol(self) -> "Scalar":
        return Scalar(self.sds_context, "ncol", [self])

    def __getitem__(self, i):
        if isinstance(i, tuple) and len(i) > 2:
            raise ValueError("Maximum of two dimensions are allowed")
        elif isinstance(i, list):
            check_no_less_than_zero(i)
            slice = self.sds_context.from_numpy(np.array(i)) + 1
            select = Matrix(self.sds_context, "table", [slice, 1, self.nRow(), 1])
            ret = Matrix(
                self.sds_context,
                "removeEmpty",
                [],
                {"target": self, "margin": '"rows"', "select": select},
            )
            return ret
        elif isinstance(i, tuple) and isinstance(i[0], list) and isinstance(i[1], list):
            raise NotImplementedError("double slicing is not supported yet")
        elif (
            isinstance(i, tuple)
            and check_is_empty_slice(i[0])
            and isinstance(i[1], list)
        ):
            check_no_less_than_zero(i[1])
            slice = self.sds_context.from_numpy(np.array(i[1])) + 1
            select = Matrix(self.sds_context, "table", [slice, 1, self.nCol(), 1])
            ret = Matrix(
                self.sds_context,
                "removeEmpty",
                [],
                {"target": self, "margin": '"cols"', "select": select},
            )
            return ret
        else:
            sliceIns = get_slice_string(i)
            return Matrix(self.sds_context, "", [self, sliceIns], brackets=True)

    def sum(self, axis: int = None) -> "OperationNode":
        """Calculate sum of matrix.

        :param axis: can be 0 or 1 to do either row or column sums
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, "colSums", [self])
        elif axis == 1:
            return Matrix(self.sds_context, "rowSums", [self])
        elif axis is None:
            return Scalar(self.sds_context, "sum", [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
        )

    def prod(self, axis: int = None) -> "OperationNode":
        """Calculate product of cells in matrix.

        :param axis: can be 0 or 1 to do either row or column sums
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, "colProds", [self])
        elif axis == 1:
            return Matrix(self.sds_context, "rowProds", [self])
        elif axis is None:
            return Scalar(self.sds_context, "prod", [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
        )

    def mean(self, axis: int = None) -> "OperationNode":
        """Calculate mean of matrix.

        :param axis: can be 0 or 1 to do either row or column means
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, "colMeans", [self])
        elif axis == 1:
            return Matrix(self.sds_context, "rowMeans", [self])
        elif axis is None:
            return Scalar(self.sds_context, "mean", [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
        )

    def max(self, axis: int = None) -> "OperationNode":
        """Calculate max of matrix.

        :param axis: can be 0 or 1 to do either row or column aggregation
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, "colMaxs", [self])
        elif axis == 1:
            return Matrix(self.sds_context, "rowMaxs", [self])
        elif axis is None:
            return Scalar(self.sds_context, "max", [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
        )

    def min(self, axis: int = None) -> "OperationNode":
        """Calculate max of matrix.

        :param axis: can be 0 or 1 to do either row or column aggregation
        :return: `Matrix` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, "colMins", [self])
        elif axis == 1:
            return Matrix(self.sds_context, "rowMins", [self])
        elif axis is None:
            return Scalar(self.sds_context, "min", [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
        )

    def countDistinct(self, axis: int = None) -> "OperationNode":
        """Calculate the number of distinct values of matrix.

        :param axis: can be 0 or 1 to do either row or column aggregation
        :return: `OperationNode` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, "colCountDistinct", [self])
        elif axis == 1:
            return Matrix(self.sds_context, "rowCountDistinct", [self])
        elif axis is None:
            return Scalar(self.sds_context, "countDistinct", [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
        )

    def countDistinctApprox(self, axis: int = None) -> "OperationNode":
        """Calculate the approximate number of distinct values of matrix.
        :param axis: can be 0 or 1 to do either row or column aggregation
        :return: `OperationNode` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, "colCountDistinctApprox", [self])
        elif axis == 1:
            return Matrix(self.sds_context, "rowCountDistinctApprox", [self])
        elif axis is None:
            return Scalar(self.sds_context, "countDistinctApprox", [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
        )

    def var(self, axis: int = None) -> "OperationNode":
        """Calculate variance of matrix.

        :param axis: can be 0 or 1 to do either row or column vars
        :return: `OperationNode` representing operation
        """
        if axis == 0:
            return Matrix(self.sds_context, "colVars", [self])
        elif axis == 1:
            return Matrix(self.sds_context, "rowVars", [self])
        elif axis is None:
            return Scalar(self.sds_context, "var", [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
        )

    def trace(self) -> "Scalar":
        """Calculate trace.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "trace", [self])

    def unique(self, axis: int = None) -> "Matrix":
        """Returns the unique values for the complete matrix, for each row or for each column.

        :param axis: can be 0 or 1 to do either row or column uniques
        :return: `Matrix` representing operation
        """
        if axis == 0:
            named_input_nodes = {"dir": '"c"'}
            return Matrix(
                self.sds_context, "unique", [self], named_input_nodes=named_input_nodes
            )
        elif axis == 1:
            named_input_nodes = {"dir": '"r"'}
            return Matrix(
                self.sds_context, "unique", [self], named_input_nodes=named_input_nodes
            )
        elif axis is None:
            return Matrix(self.sds_context, "unique", [self])
        raise ValueError(
            f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
        )

    def sd(self) -> "Scalar":
        """Calculate standard deviation of matrix.

        :return: `Scalar` representing operation
        """
        return Scalar(self.sds_context, "sd", [self])

    def abs(self) -> "Matrix":
        """Calculate absolute.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "abs", [self])

    def sqrt(self) -> "Matrix":
        """Calculate square root.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "sqrt", [self])

    def exp(self) -> "Matrix":
        """Calculate exponential.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "exp", [self])

    def floor(self) -> "Matrix":
        """Return the floor of the input, element-wise.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "floor", [self])

    def ceil(self) -> "Matrix":
        """Return the ceiling of the input, element-wise.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "ceil", [self])

    def log(self) -> "Matrix":
        """Calculate logarithm.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "log", [self])

    def sign(self) -> "Matrix":
        """Returns a matrix representing the signs of the input matrix elements,
        where 1 represents positive, 0 represents zero, and -1 represents negative.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "sign", [self])

    def sin(self) -> "Matrix":
        """Calculate sin.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "sin", [self])

    def cos(self) -> "Matrix":
        """Calculate cos.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "cos", [self])

    def tan(self) -> "Matrix":
        """Calculate tan.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "tan", [self])

    def asin(self) -> "Matrix":
        """Calculate arcsin.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "asin", [self])

    def acos(self) -> "Matrix":
        """Calculate arccos.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "acos", [self])

    def atan(self) -> "Matrix":
        """Calculate arctan.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "atan", [self])

    def sinh(self) -> "Matrix":
        """Calculate sin.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "sinh", [self])

    def cosh(self) -> "Matrix":
        """Calculate cos.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "cosh", [self])

    def tanh(self) -> "Matrix":
        """Calculate tan.

        :return: `Matrix` representing operation
        """
        return Matrix(self.sds_context, "tanh", [self])

    def moment(self, moment: int, weights: OperationNode = None) -> "Matrix":
        unnamed_inputs = [self]
        if weights is not None:
            unnamed_inputs.append(weights)
        unnamed_inputs.append(moment)
        return Matrix(self.sds_context, "moment", unnamed_inputs)

    def cholesky(self, safe: bool = False) -> "Matrix":
        """Computes the Cholesky decomposition of a symmetric, positive definite matrix

        :param safe: default value is False, if flag is True additional checks to ensure
            that the matrix is symmetric positive definite are applied, if False, checks will be skipped
        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, "cholesky", [self])

    def diag(self) -> "Matrix":
        """Create diagonal matrix from (n x 1) matrix, or take diagonal from square matrix

        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, "diag", [self])

    def svd(self) -> "Matrix":
        """
        Singular Value Decomposition of a matrix A (of size m x m), which decomposes into three matrices
        U, V, and S as A = U %% S %% t(V), where U is an m x m unitary matrix (i.e., orthogonal),
        V is an n x n unitary matrix (also orthogonal),
        and S is an m x n matrix with non-negative real numbers on the diagonal.

        matrices U <(m x m)>, S <(m x n)>, and V <(n x n)>

        :return: The MultiReturn node containing the three Matrices U,S, and V
        """

        U = Matrix(self.sds_context, "")
        S = Matrix(self.sds_context, "")
        V = Matrix(self.sds_context, "")
        output_nodes = [U, S, V]

        op = MultiReturn(
            self.sds_context, "svd", output_nodes, unnamed_input_nodes=[self]
        )
        return op

    def eigen(self) -> "Matrix":
        """
        Computes Eigen decomposition of input matrix A. The Eigen decomposition consists of
        two matrices V and w such that A = V %*% diag(w) %*% t(V). The columns of V are the
        eigenvectors of the original matrix A. And, the eigen values are given by w.
        It is important to note that this function can operate only on small-to-medium sized
        input matrix that can fit in the main memory. For larger matrices, an out-of-memory
        exception is raised.

        This function returns two matrices w and V, where w is (m x 1) and V is of size (m x m).

        :return: The MultiReturn node containing the two Matrices w and V
        """

        V = Matrix(self.sds_context, "")
        w = Matrix(self.sds_context, "")
        output_nodes = [w, V]
        op = MultiReturn(
            self.sds_context, "eigen", output_nodes, unnamed_input_nodes=[self]
        )
        return op

    def to_one_hot(self, num_classes: int) -> "Matrix":
        """OneHot encode the matrix.

        It is assumed that there is only one column to encode, and all values are whole numbers > 0

        :param num_classes: The number of classes to encode into. max value contained in the matrix must be <= num_classes
        :return: The OperationNode containing the oneHotEncoded values
        """
        if num_classes < 2:
            raise ValueError("Number of classes should be larger than 1")

        named_input_nodes = {"X": self, "numClasses": num_classes}
        return Matrix(self.sds_context, "toOneHot", named_input_nodes=named_input_nodes)

    def rbind(self, other) -> "Matrix":
        """
        Row-wise matrix concatenation, by concatenating the second matrix as additional rows to the first matrix.
        :param: The other matrix to bind to the right hand side
        :return: The OperationNode containing the concatenated matrices/frames.
        """
        return Matrix(self.sds_context, "rbind", [self, other])

    def cbind(self, other) -> "Matrix":
        """
        Column-wise matrix concatenation, by concatenating the second matrix as additional columns to the first matrix.
        :param: The other matrix to bind to the right hand side.
        :return: The OperationNode containing the concatenated matrices/frames.
        """
        return Matrix(self.sds_context, "cbind", [self, other])

    def t(self) -> "Matrix":
        """Transposes the input

        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, "t", [self])

    def order(
        self, by: int = 1, decreasing: bool = False, index_return: bool = False
    ) -> "Matrix":
        """Sort by a column of the matrix X in increasing/decreasing order and returns either the index or data

        :param by: sort matrix by this column number
        :param decreasing: If true the matrix will be sorted in decreasing order
        :param index_return: If true, the index numbers will be returned
        :return: the OperationNode representing this operation
        """

        named_input_nodes = {
            "target": self,
            "by": by,
            "decreasing": str(decreasing).upper(),
            "index.return": str(index_return).upper(),
        }

        return Matrix(
            self.sds_context, "order", [], named_input_nodes=named_input_nodes
        )

    def to_string(self, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> "Scalar":
        """Converts the input to a string representation.
        :return: `Scalar` containing the string.
        """
        return Scalar(self.sds_context, "toString", [self], kwargs)

    def isNA(self) -> "Matrix":
        """Computes a boolean indicator matrix of the same shape as the input, indicating where NA (not available)
        values are located. Currently, NA is only capturing NaN values.

        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, "isNA", [self])

    def isNaN(self) -> "Matrix":
        """Computes a boolean indicator matrix of the same shape as the input, indicating where NaN (not a number)
        values are located.

        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, "isNaN", [self])

    def isInf(self) -> "Matrix":
        """Computes a boolean indicator matrix of the same shape as the input, indicating where Inf (positive or
        negative infinity) values are located.
        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, "isInf", [self])

    def rev(self) -> "Matrix":
        """Reverses the rows

        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, "rev", [self])

    def roll(self, shift: int) -> "Matrix":
        """Reverses the rows

        :return: the OperationNode representing this operation
        """
        return Matrix(self.sds_context, "roll", [self, shift])

    def round(self) -> "Matrix":
        """round all values to nearest natural number

        :return: The Matrix representing the result of this operation
        """
        return Matrix(self.sds_context, "round", [self])

    def replace(
        self, pattern: VALID_INPUT_TYPES, replacement: VALID_INPUT_TYPES
    ) -> "Matrix":
        """
        Replace all values with replacement value
        """
        return Matrix(
            self.sds_context,
            "replace",
            named_input_nodes={
                "target": self,
                "pattern": pattern,
                "replacement": replacement,
            },
        )

    def inv(self) -> "Matrix":
        """Computes the inverse of a squared matrix.

        :return: The Matrix representing the result of this operation
        """
        return Matrix(self.sds_context, "inv", [self])

    def cumsum(self) -> "Matrix":
        """Column prefix-sum. (For row-prefix sum, use X.t().cumsum().t())

        :return: The Matrix representing the result of this operation
        """
        return Matrix(self.sds_context, "cumsum", [self])

    def cumprod(self) -> "Matrix":
        """Column prefix-product. (For row-prefix prod, use X.t().cumprod().t())

        :return: The Matrix representing the result of this operation
        """
        return Matrix(self.sds_context, "cumprod", [self])

    def cumsumprod(self) -> "Matrix":
        """Column prefix-sumprod of an 2-column matrix:
        Y = X.comsumprod(),
        where Y[i,1] = X[i,1] + X[i,2]*Y[i-1,1] for i in [1,2, .., nrow(X)]
        The aggregator is initialized with 0 (Y[0,1] = 0)

        :return: The Matrix representing the result of this operation
        """
        return Matrix(self.sds_context, "cumsumprod", [self])

    def cummin(self) -> "Matrix":
        """Column prefix-min. (For row-prefix min, use X.t().cummin().t())

        :return: The Matrix representing the result of this operation
        """
        return Matrix(self.sds_context, "cummin", [self])

    def cummax(self) -> "Matrix":
        """Column prefix-max. (For row-prefix max, use X.t().cummax().t())

        :return: The Matrix representing the result of this operation
        """
        return Matrix(self.sds_context, "cummax", [self])

    def qr(self) -> "MultiReturn":
        """Computes QR decomposition of a matrix A using Householder reflectors. The QR decomposition of A consists
        of two matrices Q and R such that A = Q%*%R where Q is an orthogonal matrix (i.e., Q%*%t(Q) = t(Q)%*%Q = I,
        identity matrix) and R is an upper triangular matrix. For efficiency purposes, this function returns the matrix
        of Householder reflector vectors H instead of Q (which is a large m x m potentially dense matrix). The Q
        matrix can be explicitly computed from H, if needed. In most applications of QR, one is interested in
        calculating Q %*% B or t(Q) %*% B – and, both can be computed directly using H instead of explicitly
        constructing the large Q matrix.

        :return: The MultiReturn node containing the two Matrices h and r
        """
        h = Matrix(self.sds_context, "")
        r = Matrix(self.sds_context, "")
        output_nodes = [h, r]

        op = MultiReturn(
            self.sds_context, "qr", output_nodes, unnamed_input_nodes=[self]
        )
        return op

    def lu(self) -> "MultiReturn":
        """Computes Pivoted LU decomposition a square matrix A. The LU decomposition consists
        of three matrices P, L, and U such that P %*% A = L %*% U, where P is a permutation matrix that is used to
        rearrange the rows in A before the decomposition can be computed. L is a lower-triangular matrix whereas U is
        an upper-triangular matrix.

        :return: The MultiReturn node containing the three Matrices p, l and u
        """
        p = Matrix(self.sds_context, "")
        l = Matrix(self.sds_context, "")
        u = Matrix(self.sds_context, "")
        output_nodes = [p, l, u]

        op = MultiReturn(
            self.sds_context, "lu", output_nodes, unnamed_input_nodes=[self]
        )
        return op

    def median(self, weights: "Matrix" = None) -> "Scalar":
        """Calculate median of a column matrix.

        :return: `Scalar` representing operation
        """
        if weights is None:
            return Scalar(self.sds_context, "median", [self])
        else:
            return Scalar(self.sds_context, "median", [self, weights])

    def quantile(self, p, weights: "Matrix" = None) -> "OperationNode":
        """Returns a column matrix with list of all quantiles requested in P.

        :param p: float for a single quantile or column matrix of requested quantiles
        :param weights: (optional) weights matrix of the same shape as self
        :return: `Matrix` or 'Scalar' representing operation
        """
        if weights is None:
            input_nodes = [self, p]
        else:
            input_nodes = [self, weights, p]

        if isinstance(p, Matrix):
            return Matrix(self.sds_context, "quantile", input_nodes)
        elif isinstance(p, float):
            if 0.0 <= p <= 1.0:
                input_nodes[-1] = self.sds_context.scalar(input_nodes[-1])
            else:
                raise ValueError("Quantile has to be between 0 and 1")
            return Scalar(self.sds_context, "quantile", input_nodes)
        else:
            raise ValueError("P has to be a Scalar or Matrix")

    def fft(self) -> "MultiReturn":
        """
        Performs the Fast Fourier Transform (FFT) on the matrix.
        :return: A MultiReturn object representing the real and imaginary parts of the FFT output.
        """

        real_output = Matrix(self.sds_context, "")
        imag_output = Matrix(self.sds_context, "")

        fft_node = MultiReturn(
            self.sds_context, "fft", [real_output, imag_output], [self]
        )

        return fft_node

    def ifft(self, imag_input: "Matrix" = None) -> "MultiReturn":
        """
        Performs the Inverse Fast Fourier Transform (IFFT) on a complex matrix.

        :param imag_input: The imaginary part of the input matrix (optional).
        :return: A MultiReturn object representing the real and imaginary parts of the IFFT output.
        """

        real_output = Matrix(self.sds_context, "")
        imag_output = Matrix(self.sds_context, "")

        if imag_input is None:
            ifft_node = MultiReturn(
                self.sds_context, "ifft", [real_output, imag_output], [self]
            )
        else:
            ifft_node = MultiReturn(
                self.sds_context, "ifft", [real_output, imag_output], [self, imag_input]
            )

        return ifft_node

    def triu(self, include_diagonal=True, return_values=True) -> "Matrix":
        """Selects the upper triangular part of a matrix, configurable to include the diagonal and return values or ones

        :param include_diagonal: boolean, default True
        :param return_values: boolean, default True, if set to False returns ones
        :return: `Matrix`
        """
        named_input_nodes = {
            "target": self,
            "diag": self.sds_context.scalar(include_diagonal),
            "values": self.sds_context.scalar(return_values),
        }
        return Matrix(
            self.sds_context, "upper.tri", named_input_nodes=named_input_nodes
        )

    def tril(self, include_diagonal=True, return_values=True) -> "Matrix":
        """Selects the lower triangular part of a matrix, configurable to include the diagonal and return values or ones

        :param include_diagonal: boolean, default True
        :param return_values: boolean, default True, if set to False returns ones
        :return: `Matrix`
        """
        named_input_nodes = {
            "target": self,
            "diag": self.sds_context.scalar(include_diagonal),
            "values": self.sds_context.scalar(return_values),
        }
        return Matrix(
            self.sds_context, "lower.tri", named_input_nodes=named_input_nodes
        )

    def argmin(self, axis: int = None) -> "OperationNode":
        """Return the index of the minimum if axis is None or a column vector for row-wise / column-wise minima
        computation.

        :param axis: can be 0 or 1 to do either row or column sums
        :return: `Matrix` representing operation for row / columns or 'Scalar' representing operation for complete
        """
        if axis == 0:
            return Matrix(self.sds_context, "rowIndexMin", [self.t()])
        elif axis == 1:
            return Matrix(self.sds_context, "rowIndexMin", [self])
        elif axis is None:
            return Matrix(
                self.sds_context,
                "rowIndexMin",
                [self.reshape(1, self.nCol() * self.nRow())],
            ).to_scalar()
        else:
            raise ValueError(
                f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
            )

    def argmax(self, axis: int = None) -> "OperationNode":
        """Return the index of the maximum if axis is None or a column vector for row-wise / column-wise maxima
        computation.

        :param axis: can be 0 or 1 to do either row or column sums
        :return: `Matrix` representing operation for row / columns or 'Scalar' representing operation for complete
        """
        if axis == 0:
            return Matrix(self.sds_context, "rowIndexMax", [self.t()])
        elif axis == 1:
            return Matrix(self.sds_context, "rowIndexMax", [self])
        elif axis is None:
            return Matrix(
                self.sds_context,
                "rowIndexMax",
                [self.reshape(1, self.nCol() * self.nRow())],
            ).to_scalar()
        else:
            raise ValueError(
                f"Axis has to be either 0, 1 or None, for column, row or complete {self.operation}"
            )

    def reshape(self, rows, cols=1):
        """Gives a new shape to a matrix without changing its data.

        :param rows: number of rows
        :param cols: number of columns, defaults to 1
        :return: `Matrix` representing operation"""
        return Matrix(self.sds_context, "matrix", [self, rows, cols])

    def __str__(self):
        return "MatrixNode"
