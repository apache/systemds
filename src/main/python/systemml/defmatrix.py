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

trigFn = [ 'exp', 'log', 'abs', 'sqrt', 'round', 'floor', 'ceil', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sign' ]
__all__ = [ 'setSparkContext', 'matrix', 'eval', 'solve' ] + trigFn


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
    matrix.sc = sc
    matrix.ml = MLContext(matrix.sc)

def checkIfMLContextIsSet():
    if matrix.ml is None:
        raise Exception('Expected setSparkContext(sc) to be called.')

########################## AST related operations ##################################

class DMLOp(object):
    """
    Represents an intermediate node of Abstract syntax tree created to generate the PyDML script
    """
    def __init__(self, inputs, dml=None):
        self.inputs = inputs
        self.dml = dml

    def _visit(self, execute=True):
        matrix.dml = matrix.dml + self.dml
    
# Special object used internally to specify the placeholder which will be replaced by output ID
# This helps to provide dml containing output ID in constructIntermediateNode
OUTPUT_ID = '$$OutputID$$'

def constructIntermediateNode(inputs, dml):
    """
    Convenient utility to create an intermediate node of AST.
    
    Parameters
    ----------
    inputs = list of input matrix objects and/or DMLOp
    dml = list of DML string (which will be eventually joined before execution). To specify out.ID, please use the placeholder  
    """
    dmlOp = DMLOp(inputs)
    out = matrix(None, op=dmlOp)
    dmlOp.dml = [out.ID if x==OUTPUT_ID else x for x in dml]
    return out

def reset():
    """
    Resets the visited status of matrix and the operators in the generated AST.
    """
    for m in matrix.visited:
        m.visited = False
    matrix.visited = []
    matrix.ml = MLContext(matrix.sc)
    matrix.dml = []
    matrix.script = pydml('')

def performDFS(outputs, execute):
    """
    Traverses the forest of nodes rooted at outputs nodes and returns the DML script to execute
    """
    for m in outputs:
        m.output = True
        m._visit(execute=execute)
    return ''.join(matrix.dml)

###############################################################################


########################## Utility functions ##################################


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
    return constructIntermediateNode(inputs, [OUTPUT_ID, ' = ', lhsStr, opStr, rhsStr, '\n'])

def getValue(obj):
    if isinstance(obj, matrix):
        return obj.ID
    elif isinstance(obj, float) or isinstance(obj, int):
        return str(obj)
    else:
        raise TypeError('Unsupported type for ' + s)

def binaryMatrixFunction(X, Y, fnName):
    """
    Common function called by supported PyDML built-in function that has two arguments.
    """
    return constructIntermediateNode([X, Y], [OUTPUT_ID, ' = ', fnName,'(', getValue(X), ', ', getValue(Y), ')\n'])

def unaryMatrixFunction(X, fnName):
    """
    Common function called by supported PyDML built-in function that has one argument.
    """
    return constructIntermediateNode([X], [OUTPUT_ID, ' = ', fnName,'(', getValue(X), ')\n'])

# utility function that converts 1:3 into DML string
def convertSeqToDML(s):
    ret = []
    if s is None:
        return ''
    elif isinstance(s, slice):
        if s.step is not None:
            raise ValueError('Slicing with step is not supported.')
        if s.start is None:
            ret = ret + [ '0 : ' ]
        else:
            ret = ret + [ getValue(s.start), ':' ]
        if s.start is None:
            ret = ret + [ '' ]
        else:
            ret = ret + [ getValue(s.stop) ]
    else:
        ret = ret + [ getValue(s) ]
    return ''.join(ret)

# utility function that converts index (such as [1, 2:3]) into DML string
def getIndexingDML(index):
    ret = [ '[' ]
    if isinstance(index, tuple) and len(index) == 1:
        ret = ret + [ convertSeqToDML(index[0]), ',' ]
    elif isinstance(index, tuple) and len(index) == 2:
        ret = ret + [ convertSeqToDML(index[0]), ',', convertSeqToDML(index[1]) ]
    else:
        raise TypeError('matrix indexes can only be tuple of length 2. For example: m[1,1], m[0:1,], m[:, 0:1]')
    return ret + [ ']' ]

def convertOutputsToList(outputs):
    if isinstance(outputs, matrix):
        return [ outputs ]
    elif isinstance(outputs, list):
        for o in outputs:
            if not isinstance(o, matrix):
                raise TypeError('Only matrix or list of matrix allowed')
        return outputs
    else:
        raise TypeError('Only matrix or list of matrix allowed')

def resetOutputFlag(outputs):
    for m in outputs:
        m.output = False

def populateOutputs(outputs, results, outputDF):
    """
    Set the attribute 'data' of the matrix by fetching it from MLResults class
    """
    for m in outputs:
        if outputDF:
            m.data = results.getDataFrame(m.ID)
        else:
            m.data = results.getNumPyArray(m.ID)

###############################################################################

########################## Global user-facing functions #######################

def exp(X):
    return unaryMatrixFunction(X, 'exp')
    
def log(X, y=None):
    if y is None:
        return unaryMatrixFunction(X, 'log')
    else:
        return binaryMatrixFunction(X, y, 'log')

def abs(X):
    return unaryMatrixFunction(X, 'abs')
        
def sqrt(X):
    return unaryMatrixFunction(X, 'sqrt')
    
def round(X):
    return unaryMatrixFunction(X, 'round')

def floor(X):
    return unaryMatrixFunction(X, 'floor')

def ceil(X):
    return unaryMatrixFunction(X, 'ceil')
    
def sin(X):
    return unaryMatrixFunction(X, 'sin')

def cos(X):
    return unaryMatrixFunction(X, 'cos')

def tan(X):
    return unaryMatrixFunction(X, 'tan')

def asin(X):
    return unaryMatrixFunction(X, 'asin')

def acos(X):
    return unaryMatrixFunction(X, 'acos')

def atan(X):
    return unaryMatrixFunction(X, 'atan')

def sign(X):
    return unaryMatrixFunction(X, 'sign')
    
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
    outputs: list of matrices or a matrix object
    outputDF: back the data of matrix as PySpark DataFrame
    """
    checkIfMLContextIsSet()
    reset()
    outputs = convertOutputsToList(outputs)
    matrix.script.scriptString = performDFS(outputs, execute)
    if not execute:
        resetOutputFlag(outputs)
        return matrix.script.scriptString
    results = matrix.ml.execute(matrix.script)
    populateOutputs(outputs, results, outputDF)
    resetOutputFlag(outputs)
            
###############################################################################

# DESIGN DECISIONS:
# 1. Until eval() method is invoked, we create an AST (not exposed to the user) that consist of unevaluated operations and data required by those operations.
#    As an anology, a spark user can treat eval() method similar to calling RDD.persist() followed by RDD.count().  
# 2. The AST consist of two kinds of nodes: either of type matrix or of type DMLOp.
#    Both these classes expose _visit method, that helps in traversing the AST in DFS manner.
# 3. A matrix object can either be evaluated or not. 
#    If evaluated, the attribute 'data' is set to one of the supported types (for example: NumPy array or DataFrame). In this case, the attribute 'op' is set to None.
#    If not evaluated, the attribute 'op' which refers to one of the intermediate node of AST and if of type DMLOp.  In this case, the attribute 'data' is set to None.
# 5. DMLOp has an attribute 'inputs' which contains list of matrix objects or DMLOp. 
# 6. To simplify the traversal, every matrix object is considered immutable and an matrix operations creates a new matrix object.
#    As an example: 
#    - m1 = sml.matrix(np.ones((3,3))) creates a matrix object backed by 'data=(np.ones((3,3))'.
#    - m1 = m1 * 2 will create a new matrix object which is now backed by 'op=DMLOp( ... )' whose input is earlier created matrix object.
# 7. Left indexing (implemented in __setitem__ method) is a special case, where Python expects the existing object to be mutated.
#    To ensure the above property, we make deep copy of existing object and point any references to the left-indexed matrix to the newly created object.
#    Then the left-indexed matrix is set to be backed by DMLOp consisting of following pydml:
#    left-indexed-matrix = new-deep-copied-matrix
#    left-indexed-matrix[index] = value 
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
        self.referenced = []
        # op refers to the node of Abstract Syntax Tree created internally for lazy evaluation
        self.op = op
        self.data = data
        if not (isinstance(data, SUPPORTED_TYPES) or hasattr(data, '_jdf') or (data is None and op is not None)):
            raise TypeError('Unsupported input type')
        if op is not None:
            self.referenced = self.referenced + [ op ]

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

    def _markAsVisited(self):
        self.visited = True
        # for cleanup
        matrix.visited = matrix.visited + [ self ]
        return self
        
    def _registerAsInput(self, execute):
        # TODO: Remove this when automatic registration of frame is resolved
        matrix.dml = [ self.ID,  ' = load(\" \", format=\"csv\")\n'] + matrix.dml
        if isinstance(self.data, DataFrame) and execute:
            matrix.script.input(self.ID, self.data)
        elif execute:
            matrix.script.input(self.ID, convertToMatrixBlock(matrix.sc, self.data))
        return self
        
    def _registerAsOutput(self, execute):
        # TODO: Remove this when automatic registration of frame is resolved
        matrix.dml = matrix.dml + ['save(',  self.ID, ', \" \")\n']
        if execute:
            matrix.script.output(self.ID)
    
    def _visit(self, execute=True):
        """
        This function is called for two scenarios:
        1. For printing the PyDML script which has not yet been evaluated (execute=False). See '__repr__' method.
        2. Called as part of 'eval' method (execute=True). In this scenario, it builds the PyDML script by visiting itself
        and its child nodes. Also, it does appropriate registration as input or output that is required by MLContext.
        """
        if self.visited:
            return self
        self._markAsVisited()
        if self.data is not None:
            self._registerAsInput(execute)
        elif self.op is not None:
            # Traverse the AST
            for m in self.op.inputs:
                m._visit(execute=execute)
            self.op._visit(execute=execute)
        else:
            raise Exception('Expected either op or data to be set')
        if self.data is None and self.output:
            self._registerAsOutput(execute)
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

    ######################### Arithmetic operators ######################################
    
    def __add__(self, other):
        return binaryOp(self, other, ' + ')

    def __sub__(self, other):
        return binaryOp(self, other, ' - ')

    def __mul__(self, other):
        return binaryOp(self, other, ' * ')

    def __floordiv__(self, other):
        return binaryOp(self, other, ' // ')

    def __div__(self, other):
        """
        Performs division (Python 2 way).
        """
        return binaryOp(self, other, ' / ')

    def __truediv__(self, other):
        """
        Performs division (Python 3 way).
        """
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
        
    def dot(self, other):
        """
        Numpy way of performing matrix multiplication
        """
        return binaryMatrixFunction(self, other, 'dot')
    
    def __matmul__(self, other):
        """
        Performs matrix multiplication (infix operator: @). See PEP 465)
        """
        return binaryMatrixFunction(self, other, 'dot')
    
    
    ######################### Relational/Boolean operators ######################################
    
    def __lt__(self, other):
        return binaryOp(other, self, ' < ')
        
    def __le__(self, other):
        return binaryOp(other, self, ' <= ')
    
    def __gt__(self, other):
        return binaryOp(other, self, ' > ')
        
    def __ge__(self, other):
        return binaryOp(other, self, ' >= ')
        
    def __eq__(self, other):
        return binaryOp(other, self, ' == ')
        
    def __ne__(self, other):
        return binaryOp(other, self, ' != ')
        
    def __and__(self, other):
        return binaryOp(other, self, ' & ')

    def __or__(self, other):
        return binaryOp(other, self, ' | ')
        
    ######################### Aggregation functions ######################################
    
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
        
    ######################### Indexing operators ######################################
    
    def __getitem__(self, index):
        """
        Implements evaluation of right indexing operations such as m[1,1], m[0:1,], m[:, 0:1]
        """
        dmlOp = DMLOp([self])
        out = matrix(None, op=dmlOp)
        dmlOp.dml = [out.ID, ' = ', self.ID ] + getIndexingDML(index) + [ '\n' ]
        return out
    
    # Performs deep copy if the matrix is backed by data
    def _prepareForInPlaceUpdate(self):
        temp = matrix(self.data, op=self.op)
        self.ID, temp.ID = temp.ID, self.ID # Copy even the IDs as the IDs might be used to create DML
        for op in self.referenced:
            op.inputs.remove(self) #while self in op.inputs:
            op.inputs = op.inputs + [ temp ]
        self.op = DMLOp([temp], dml=[self.ID, " = ", temp.ID])
        self.data = None       
    
    def __setitem__(self, index, value):
        """
        Implements evaluation of left indexing operations such as m[1,1]=2
        """
        self._prepareForInPlaceUpdate()
        if isinstance(value, matrix): 
            self.op.inputs = self.op.inputs + [ value ]
        self.op.dml = self.op.dml + [ '\n', self.ID ] + getIndexingDML(index) + [ ' = ',  getValue(value), '\n']