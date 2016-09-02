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

__all__ = [ 'setSparkContext', 'matrix', 'eval', 'solve']

from pyspark import SparkContext
from pyspark.sql import DataFrame, SQLContext

from . import MLContext, pydml
from .converters import *

def setSparkContext(sc):
    """
    Before using the matrix, the user needs to invoke this function.

    Parameters
    ----------
    sc: SparkContext
        SparkContext
    """
    matrix.ml = MLContext(sc)
    matrix.sc = sc


def checkIfMLContextIsSet():
    if matrix.ml is None:
        raise Exception('Expected setSparkContext(sc) to be called.')


class DMLOp(object):
    """
    Represents an intermediate node of Abstract syntax tree created to generate the PyDML script
    """
    def __init__(self, inputs, dml=None):
        self.inputs = inputs
        self.dml = dml

    def _visit(self, execute=True):
        matrix.dml = matrix.dml + self.dml


def reset():
    """
    Resets the visited status of matrix and the operators in the generated AST.
    """
    for m in matrix.visited:
        m.visited = False
    matrix.visited = []


def binaryOp(lhs, rhs, opStr):
    """
    Common function called by all the binary operators in matrix class
    """
    inputs = []
    if isinstance(lhs, matrix):
        lhsStr = lhs.ID
        inputs = [lhs]
    elif isinstance(lhs, float) or isinstance(lhs, int):
        lhsStr = str(lhs)
    else:
        raise TypeError('Incorrect type')
    if isinstance(rhs, matrix):
        rhsStr = rhs.ID
        inputs = inputs + [rhs]
    elif isinstance(rhs, float) or isinstance(rhs, int):
        rhsStr = str(rhs)
    else:
        raise TypeError('Incorrect type')
    dmlOp = DMLOp(inputs)
    out = matrix(None, op=dmlOp)
    dmlOp.dml = [out.ID, ' = ', lhsStr, opStr, rhsStr, '\n']
    return out


def binaryMatrixFunction(X, Y, fnName):
    """
    Common function called by supported PyDML built-in function that has two arguments both of which are matrices.
    TODO: This needs to be generalized to support arbitrary arguments of differen types.
    """
    if not isinstance(X, matrix) or not isinstance(Y, matrix):
        raise TypeError('Incorrect input type. Expected matrix type')
    inputs = [X, Y]
    dmlOp = DMLOp(inputs)
    out = matrix(None, op=dmlOp)
    dmlOp.dml = [out.ID, ' = ', fnName,'(', X.ID, ', ', Y.ID, ')\n']
    return out


def solve(A, b):
    """
    Computes the least squares solution for system of linear equations A %*% x = b

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> import SystemML as sml
    >>> from pyspark.sql import SQLContext
    >>> diabetes = datasets.load_diabetes()
    >>> diabetes_X = diabetes.data[:, np.newaxis, 2]
    >>> X_train = diabetes_X[:-20]
    >>> X_test = diabetes_X[-20:]
    >>> y_train = diabetes.target[:-20]
    >>> y_test = diabetes.target[-20:]
    >>> sml.setSparkContext(sc)
    >>> X = sml.matrix(X_train)
    >>> y = sml.matrix(y_train)
    >>> A = X.transpose().dot(X)
    >>> b = X.transpose().dot(y)
    >>> beta = sml.solve(A, b).toNumPyArray()
    >>> y_predicted = X_test.dot(beta)
    >>> print('Residual sum of squares: %.2f' % np.mean((y_predicted - y_test) ** 2))
    Residual sum of squares: 25282.12
    """
    return binaryMatrixFunction(A, b, 'solve')


def eval(outputs, outputDF=False, execute=True):
    """
    Executes the unevaluated DML script and computes the matrices specified by outputs.

    Parameters
    ----------
    outputs: list of matrices
    outputDF: back the data of matrix as PySpark DataFrame
    """
    checkIfMLContextIsSet()
    reset()
    matrix.dml = []
    matrix.script = pydml('')
    if isinstance(outputs, matrix):
        outputs = [ outputs ]
    elif not isinstance(outputs, list):
        raise TypeError('Incorrect input type')
    for m in outputs:
        m.output = True
        m._visit(execute=execute)
    if not execute:
        return ''.join(matrix.dml)
    matrix.script.scriptString = ''.join(matrix.dml)
    results = matrix.ml.execute(matrix.script)
    # Note: an evaluated matrix contains a data field computed by eval method as DataFrame or NumPy array.
    for m in outputs:
        if outputDF:
            m.data = results.getDataFrame(m.ID)
        else:
            m.data = results.getNumPyArray(m.ID)


class matrix(object):
    """
    matrix class is a python wrapper that implements basic matrix operator.
    Note: an evaluated matrix contains a data field computed by eval method as DataFrame or NumPy array.

    Examples
    --------
    >>> import SystemML as sml
    >>> import numpy as np
    >>> sml.setSparkContext(sc)

    Welcome to Apache SystemML!

    >>> m1 = sml.matrix(np.ones((3,3)) + 2)
    >>> m2 = sml.matrix(np.ones((3,3)) + 3)
    >>> m2 = m1 * (m2 + m1)
    >>> m4 = 1.0 - m2
    >>> m4
    # This matrix (mVar5) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPyArray() or toDataFrame() or toPandas() methods.
    mVar1 = load(" ", format="csv")
    mVar2 = load(" ", format="csv")
    mVar3 = mVar2 + mVar1
    mVar4 = mVar1 * mVar3
    mVar5 = 1.0 - mVar4
    save(mVar5, " ")

    <SystemML.defmatrix.matrix object>
    >>> m2.eval()
    >>> m2
    # This matrix (mVar4) is backed by NumPy array. To fetch the NumPy array, invoke toNumPyArray() method.
    <SystemML.defmatrix.matrix object>
    >>> m4
    # This matrix (mVar5) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPyArray() or toDataFrame() or toPandas() methods.
    mVar4 = load(" ", format="csv")
    mVar5 = 1.0 - mVar4
    save(mVar5, " ")

    <SystemML.defmatrix.matrix object>
    >>> m4.sum(axis=1).toNumPyArray()
    array([[-60.],
           [-60.],
           [-60.]])
    """
    # Global variable that is used to keep track of intermediate matrix variables in the DML script
    systemmlVarID = 0

    # Since joining of string is expensive operation, we collect the set of strings into list and then join
    # them before execution: See matrix.script.scriptString = ''.join(matrix.dml) in eval() method
    dml = []

    # Represents MLContext's script object
    script = None

    # Represents MLContext object
    ml = None

    # Contains list of nodes visited in Abstract Syntax Tree. This helps to avoid computation of matrix objects
    # that have been previously evaluated.
    visited = []

    def __init__(self, data, op=None):
        """
        Constructs a lazy matrix

        Parameters
        ----------
        data: NumPy ndarray, Pandas DataFrame, scipy sparse matrix or PySpark DataFrame. (data cannot be None for external users, 'data=None' is used internally for lazy evaluation).
        """
        checkIfMLContextIsSet()
        self.visited = False
        matrix.systemmlVarID += 1
        self.output = False
        self.ID = 'mVar' + str(matrix.systemmlVarID)
        if isinstance(data, SUPPORTED_TYPES):
            self.data = data
        elif hasattr(data, '_jdf'):
            self.data = data
        elif data is None and op is not None:
            self.data = None
            # op refers to the node of Abstract Syntax Tree created internally for lazy evaluation
            self.op = op
        else:
            raise TypeError('Unsupported input type')

    def eval(self, outputDF=False):
        """
        This is a convenience function that calls the global eval method
        """
        eval([self], outputDF=False)

    def toPandas(self):
        """
        This is a convenience function that calls the global eval method and then converts the matrix object into Pandas DataFrame.
        """
        if self.data is None:
            self.eval()
        return convertToPandasDF(self.data)

    def toNumPyArray(self):
        """
        This is a convenience function that calls the global eval method and then converts the matrix object into NumPy array.
        """
        if self.data is None:
            self.eval()
        if isinstance(self.data, DataFrame):
            self.data = self.data.toPandas().as_matrix()
        # Always keep default format as NumPy array if possible
        return self.data

    def toDataFrame(self):
        """
        This is a convenience function that calls the global eval method and then converts the matrix object into DataFrame.
        """
        if self.data is None:
            self.eval(outputDF=True)
        if not isinstance(self.data, DataFrame):
            if MLResults.sqlContext is None:
                MLResults.sqlContext = SQLContext(matrix.sc)
            self.data = sqlContext.createDataFrame(self.toPandas())
        return self.data

    def _visit(self, execute=True):
        """
        This function is called for two scenarios:
        1. For printing the PyDML script which has not yet been evaluated (execute=False). See '__repr__' method.
        2. Called as part of 'eval' method (execute=True). In this scenario, it builds the PyDML script by visiting itself
        and its child nodes. Also, it does appropriate registration as input or output that is required by MLContext.
        """
        if self.visited:
            return self
        self.visited = True
        # for cleanup
        matrix.visited = matrix.visited + [ self ]
        if self.data is not None:
            matrix.dml = matrix.dml + [ self.ID,  ' = load(\" \", format=\"csv\")\n']
            if isinstance(self.data, DataFrame) and execute:
                matrix.script.input(self.ID, self.data)
            elif  execute:
                matrix.script.input(self.ID, convertToMatrixBlock(matrix.sc, self.data))
            return self
        elif self.op is not None:
            for m in self.op.inputs:
                m._visit(execute=execute)
            self.op._visit(execute=execute)
        else:
            raise Exception('Expected either op or data to be set')
        if self.data is None and self.output:
            matrix.dml = matrix.dml + ['save(',  self.ID, ', \" \")\n']
            if execute:
                matrix.script.output(self.ID)
        return self

    def __repr__(self):
        """
        This function helps to debug matrix class and also examine the generated PyDML script
        """
        if self.data is None:
            print('# This matrix (' + self.ID + ') is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPyArray() or toDataFrame() or toPandas() methods.\n' + eval([self], execute=False))
        elif isinstance(self.data, DataFrame):
            print('# This matrix (' + self.ID + ') is backed by PySpark DataFrame. To fetch the DataFrame, invoke toDataFrame() method.')
        else:
            print('# This matrix (' + self.ID + ') is backed by NumPy array. To fetch the NumPy array, invoke toNumPyArray() method.')
        return '<SystemML.defmatrix.matrix object>'

    def __add__(self, other):
        return binaryOp(self, other, ' + ')

    def __sub__(self, other):
        return binaryOp(self, other, ' - ')

    def __mul__(self, other):
        return binaryOp(self, other, ' * ')

    def __floordiv__(self, other):
        return binaryOp(self, other, ' // ')

    def __div__(self, other):
        return binaryOp(self, other, ' / ')

    def __mod__(self, other):
        return binaryOp(self, other, ' % ')

    def __pow__(self, other):
        return binaryOp(self, other, ' ** ')

    def __radd__(self, other):
        return binaryOp(other, self, ' + ')

    def __rsub__(self, other):
        return binaryOp(other, self, ' - ')

    def __rmul__(self, other):
        return binaryOp(other, self, ' * ')

    def __rfloordiv__(self, other):
        return binaryOp(other, self, ' // ')

    def __rdiv__(self, other):
        return binaryOp(other, self, ' / ')

    def __rmod__(self, other):
        return binaryOp(other, self, ' % ')

    def __rpow__(self, other):
        return binaryOp(other, self, ' ** ')

    def sum(self, axis=None):
        return self._aggFn('sum', axis)

    def mean(self, axis=None):
        return self._aggFn('mean', axis)

    def max(self, axis=None):
        return self._aggFn('max', axis)

    def min(self, axis=None):
        return self._aggFn('min', axis)

    def argmin(self, axis=None):
        return self._aggFn('argmin', axis)

    def argmax(self, axis=None):
        return self._aggFn('argmax', axis)

    def cumsum(self, axis=None):
        return self._aggFn('cumsum', axis)

    def transpose(self, axis=None):
        return self._aggFn('transpose', axis)

    def trace(self, axis=None):
        return self._aggFn('trace', axis)

    def _aggFn(self, fnName, axis):
        """
        Common function that is called for functions that have axis as parameter.
        """
        dmlOp = DMLOp([self])
        out = matrix(None, op=dmlOp)
        if axis is None:
            dmlOp.dml = [out.ID, ' = ', fnName, '(', self.ID, ')\n']
        else:
            dmlOp.dml = [out.ID, ' = ', fnName, '(', self.ID, ', axis=', str(axis) ,')\n']
        return out

    def dot(self, other):
        return binaryMatrixFunction(self, other, 'dot')
