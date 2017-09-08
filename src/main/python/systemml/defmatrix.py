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

__all__ = [ 'setSparkContext', 'matrix', 'eval', 'solve', 'DMLOp', 'set_lazy', 'debug_array_conversion', 'load', 'full', 'seq' ]

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, spmatrix
try:
    import py4j.java_gateway
    from py4j.java_gateway import JavaObject
    from pyspark import SparkContext
    from pyspark.sql import DataFrame, SparkSession
    import pyspark.mllib.common
except ImportError:
    raise ImportError('Unable to import `pyspark`. Hint: Make sure you are running with PySpark.')

from . import MLContext, pydml, _java2py, Matrix
from .converters import *

def setSparkContext(sc):
    """
    Before using the matrix, the user needs to invoke this function if SparkContext is not previously created in the session.

    Parameters
    ----------
    sc: SparkContext
        SparkContext
    """
    matrix.sc = sc
    matrix.sparkSession = SparkSession.builder.getOrCreate()
    matrix.ml = MLContext(matrix.sc)


def check_MLContext():
    if matrix.ml is None:
        if SparkContext._active_spark_context is not None:
            setSparkContext(SparkContext._active_spark_context)
        else:
            raise Exception('Expected setSparkContext(sc) to be called, where sc is active SparkContext.')

########################## AST related operations ##################################

class DMLOp(object):
    """
    Represents an intermediate node of Abstract syntax tree created to generate the PyDML script
    """
    def __init__(self, inputs, dml=None):
        self.inputs = inputs
        self.dml = dml
        self.ID = None
        self.depth = 1
        for m in self.inputs:
            m.referenced = m.referenced + [ self ]
            if isinstance(m, matrix) and m.op is not None:
                self.depth = max(self.depth, m.op.depth + 1)

    MAX_DEPTH = 0
    
    def _visit(self, execute=True):
        matrix.dml = matrix.dml + self.dml

    def _print_ast(self, numSpaces):
        ret = []
        for m in self.inputs:
            ret = [ m._print_ast(numSpaces+2) ]
        return ''.join(ret)

# Special object used internally to specify the placeholder which will be replaced by output ID
# This helps to provide dml containing output ID in construct_intermediate_node
OUTPUT_ID = '$$OutputID$$'

def set_lazy(isLazy):
    """
    This method allows users to set whether the matrix operations should be executed in lazy manner.
    
    Parameters
    ----------
    isLazy: True if matrix operations should be evaluated in lazy manner.
    """
    if isLazy:
        DMLOp.MAX_DEPTH = 0
    else:
        DMLOp.MAX_DEPTH = 1
    
def construct_intermediate_node(inputs, dml):
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
    if DMLOp.MAX_DEPTH > 0 and out.op.depth >= DMLOp.MAX_DEPTH:
        out.eval()
    return out

def load(file, format='csv'):
    """
    Allows user to load a matrix from filesystem

    Parameters
    ----------
    file: filepath
    format: can be csv, text or binary or mm
    """
    return construct_intermediate_node([], [OUTPUT_ID, ' = load(\"', file, '\", format=\"', format, '\")\n'])

def full(shape, fill_value):
    """
    Return a new array of given shape filled with fill_value.

    Parameters
    ----------
    shape: tuple of length 2
    fill_value: float or int
    """
    return construct_intermediate_node([], [OUTPUT_ID, ' = full(', str(fill_value), ', rows=', str(shape[0]), ', cols=', str(shape[1]), ')\n'])    

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

def perform_dfs(outputs, execute):
    """
    Traverses the forest of nodes rooted at outputs nodes and returns the DML script to execute
    """
    for m in outputs:
        m.output = True
        m._visit(execute=execute)
    return ''.join(matrix.dml)

###############################################################################


########################## Utility functions ##################################

def _log_base(val, base):
    if not isinstance(val, str):
        raise ValueError('The val to _log_base should be of type string')
    return '(log(' + val + ')/log(' + str(base) + '))' 
    
def _matricize(lhs, inputs):
    """
    Utility fn to convert the supported types to matrix class or to string (if float or int)
    and return the string to be passed to DML as well as inputs
    """
    if isinstance(lhs, SUPPORTED_TYPES):
        lhs = matrix(lhs)
    if isinstance(lhs, matrix):
        lhsStr = lhs.ID
        inputs = inputs + [lhs]
    elif isinstance(lhs, float) or isinstance(lhs, int):
        lhsStr = str(lhs)
    else:
        raise TypeError('Incorrect type')
    return lhsStr, inputs
    
def binary_op(lhs, rhs, opStr):
    """
    Common function called by all the binary operators in matrix class
    """
    inputs = []
    lhsStr, inputs = _matricize(lhs, inputs)
    rhsStr, inputs = _matricize(rhs, inputs)
    return construct_intermediate_node(inputs, [OUTPUT_ID, ' = ', lhsStr, opStr, rhsStr, '\n'])

def binaryMatrixFunction(X, Y, fnName):
    """
    Common function called by supported PyDML built-in function that has two arguments.
    """
    inputs = []
    lhsStr, inputs = _matricize(X, inputs)
    rhsStr, inputs = _matricize(Y, inputs)
    return construct_intermediate_node(inputs, [OUTPUT_ID, ' = ', fnName,'(', lhsStr, ', ', rhsStr, ')\n'])

def unaryMatrixFunction(X, fnName):
    """
    Common function called by supported PyDML built-in function that has one argument.
    """
    inputs = []
    lhsStr, inputs = _matricize(X, inputs)
    return construct_intermediate_node(inputs, [OUTPUT_ID, ' = ', fnName,'(', lhsStr, ')\n'])

def seq(start=None, stop=None, step=1):
    """
    Creates a single column vector with values starting from <start>, to <stop>, in increments of <step>.
    Note: Unlike Numpy's arange which returns a row-vector, this returns a column vector.
    Also, Unlike Numpy's arange which doesnot include stop, this method includes stop in the interval.
    
    Parameters
    ----------
    start: int or float [Optional: default = 0]
    stop: int or float
    step : int float [Optional: default = 1]
    """
    if start is None and stop is None:
        raise ValueError('Both start and stop cannot be None')
    elif start is not None and stop is None:
        stop = start
        start = 0
    return construct_intermediate_node([], [OUTPUT_ID, ' = seq(', str(start), ',', str(stop), ',',  str(step), ')\n'])
    
# utility function that converts 1:3 into DML string
def convert_seq_to_dml(s):
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
        ret = ret + [ convert_seq_to_dml(index[0]), ',' ]
    elif isinstance(index, tuple) and len(index) == 2:
        ret = ret + [ convert_seq_to_dml(index[0]), ',', convert_seq_to_dml(index[1]) ]
    else:
        raise TypeError('matrix indexes can only be tuple of length 2. For example: m[1,1], m[0:1,], m[:, 0:1]')
    return ret + [ ']' ]

def convert_outputs_to_list(outputs):
    if isinstance(outputs, matrix):
        return [ outputs ]
    elif isinstance(outputs, list):
        for o in outputs:
            if not isinstance(o, matrix):
                raise TypeError('Only matrix or list of matrix allowed')
        return outputs
    else:
        raise TypeError('Only matrix or list of matrix allowed')

def reset_output_flag(outputs):
    for m in outputs:
        m.output = False
    

###############################################################################

########################## Global user-facing functions #######################

def solve(A, b):
    """
    Computes the least squares solution for system of linear equations A %*% x = b

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> import SystemML as sml
    >>> from pyspark.sql import SparkSession
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
    >>> beta = sml.solve(A, b).toNumPy()
    >>> y_predicted = X_test.dot(beta)
    >>> print('Residual sum of squares: %.2f' % np.mean((y_predicted - y_test) ** 2))
    Residual sum of squares: 25282.12
    """
    return binaryMatrixFunction(A, b, 'solve')

def eval(outputs, execute=True):
    """
    Executes the unevaluated DML script and computes the matrices specified by outputs.

    Parameters
    ----------
    outputs: list of matrices or a matrix object
    execute: specified whether to execute the unevaluated operation or just return the script.
    """
    check_MLContext()
    reset()
    outputs = convert_outputs_to_list(outputs)
    matrix.script.setScriptString(perform_dfs(outputs, execute))
    if not execute:
        reset_output_flag(outputs)
        return matrix.script.scriptString
    results = matrix.ml.execute(matrix.script)
    for m in outputs:
        m.eval_data = results._java_results.get(m.ID)
    reset_output_flag(outputs)


def debug_array_conversion(throwError):
    matrix.THROW_ARRAY_CONVERSION_ERROR = throwError
    
def _get_new_var_id():
    matrix.systemmlVarID += 1
    return 'mVar' + str(matrix.systemmlVarID)

###############################################################################

class matrix(object):
    """
    matrix class is a python wrapper that implements basic matrix operators, matrix functions
    as well as converters to common Python types (for example: Numpy arrays, PySpark DataFrame
    and Pandas DataFrame). 
    
    The operators supported are:
    
    1. Arithmetic operators: +, -, *, /, //, %, ** as well as dot (i.e. matrix multiplication)
    2. Indexing in the matrix
    3. Relational/Boolean operators: <, <=, >, >=, ==, !=, &, |
    
    In addition, following functions are supported for matrix:
    
    1. transpose
    2. Aggregation functions: sum, mean, var, sd, max, min, argmin, argmax, cumsum
    3. Global statistical built-In functions: exp, log, abs, sqrt, round, floor, ceil, sin, cos, tan, asin, acos, atan, sign, solve
    
    For all the above functions, we always return a two dimensional matrix, especially for aggregation functions with axis. 
    For example: Assuming m1 is a matrix of (3, n), NumPy returns a 1d vector of dimension (3,) for operation m1.sum(axis=1)
    whereas SystemML returns a 2d matrix of dimension (3, 1).
    
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
    # This matrix (mVar5) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPy() or toDF() or toPandas() methods.
    mVar1 = load(" ", format="csv")
    mVar2 = load(" ", format="csv")
    mVar3 = mVar2 + mVar1
    mVar4 = mVar1 * mVar3
    mVar5 = 1.0 - mVar4
    save(mVar5, " ")
    >>> m2.eval()
    >>> m2
    # This matrix (mVar4) is backed by NumPy array. To fetch the NumPy array, invoke toNumPy() method.
    >>> m4
    # This matrix (mVar5) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPy() or toDF() or toPandas() methods.
    mVar4 = load(" ", format="csv")
    mVar5 = 1.0 - mVar4
    save(mVar5, " ")
    >>> m4.sum(axis=1).toNumPy()
    array([[-60.],
           [-60.],
           [-60.]])
    
    Design Decisions:
    
    1. Until eval() method is invoked, we create an AST (not exposed to the user) that consist of unevaluated operations and data required by those operations.
       As an anology, a spark user can treat eval() method similar to calling RDD.persist() followed by RDD.count().
    2. The AST consist of two kinds of nodes: either of type matrix or of type DMLOp.
       Both these classes expose _visit method, that helps in traversing the AST in DFS manner.
    3. A matrix object can either be evaluated or not.
       If evaluated, the attribute 'data' is set to one of the supported types (for example: NumPy array or DataFrame). In this case, the attribute 'op' is set to None.
       If not evaluated, the attribute 'op' which refers to one of the intermediate node of AST and if of type DMLOp.  In this case, the attribute 'data' is set to None.
    4. DMLOp has an attribute 'inputs' which contains list of matrix objects or DMLOp.
    5. To simplify the traversal, every matrix object is considered immutable and an matrix operations creates a new matrix object.
       As an example: 
       `m1 = sml.matrix(np.ones((3,3)))` creates a matrix object backed by 'data=(np.ones((3,3))'.
       `m1 = m1 * 2` will create a new matrix object which is now backed by 'op=DMLOp( ... )' whose input is earlier created matrix object.
    6. Left indexing (implemented in __setitem__ method) is a special case, where Python expects the existing object to be mutated.
       To ensure the above property, we make deep copy of existing object and point any references to the left-indexed matrix to the newly created object.
       Then the left-indexed matrix is set to be backed by DMLOp consisting of following pydml:
       left-indexed-matrix = new-deep-copied-matrix
       left-indexed-matrix[index] = value
    7. Please use m.print_ast() and/or  type `m` for debugging. Here is a sample session:
    
       >>> npm = np.ones((3,3))
       >>> m1 = sml.matrix(npm + 3)
       >>> m2 = sml.matrix(npm + 5)
       >>> m3 = m1 + m2
       >>> m3
       mVar2 = load(" ", format="csv")
       mVar1 = load(" ", format="csv")
       mVar3 = mVar1 + mVar2
       save(mVar3, " ")
       >>> m3.print_ast()
       - [mVar3] (op).
         - [mVar1] (data).
         - [mVar2] (data).    
    
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
        self.dtype = np.double
        check_MLContext()
        self.visited = False
        self.output = False
        self.ID = _get_new_var_id()
        self.referenced = []
        # op refers to the node of Abstract Syntax Tree created internally for lazy evaluation
        self.op = op
        self.eval_data = data
        self._shape = None
        if isinstance(data, SUPPORTED_TYPES):
            self._shape = data.shape
        if not (isinstance(data, SUPPORTED_TYPES) or hasattr(data, '_jdf') or (data is None and op is not None)):
            raise TypeError('Unsupported input type')

    def eval(self):
        """
        This is a convenience function that calls the global eval method
        """
        eval([self])
        
    def toPandas(self):
        """
        This is a convenience function that calls the global eval method and then converts the matrix object into Pandas DataFrame.
        """
        self.eval()
        if isinstance(self.eval_data, py4j.java_gateway.JavaObject):
            self.eval_data = _java2py(SparkContext._active_spark_context, self.eval_data)
        if isinstance(self.eval_data, Matrix):
            self.eval_data = self.eval_data.toNumPy()
        self.eval_data = convertToPandasDF(self.eval_data)
        return self.eval_data

    def toNumPy(self):
        """
        This is a convenience function that calls the global eval method and then converts the matrix object into NumPy array.
        """
        self.eval()
        if isinstance(self.eval_data, py4j.java_gateway.JavaObject):
            self.eval_data = _java2py(SparkContext._active_spark_context, self.eval_data)
        if isinstance(self.eval_data, Matrix):
            self.eval_data = self.eval_data.toNumPy()
            return self.eval_data
        if isinstance(self.eval_data, pd.DataFrame):
            self.eval_data = self.eval_data.as_matrix()
        elif isinstance(self.eval_data, DataFrame):
            self.eval_data = self.eval_data.toPandas().as_matrix()
        elif isinstance(self.eval_data, spmatrix):
            self.eval_data = self.eval_data.toarray()
        elif isinstance(self.eval_data, Matrix):
            self.eval_data = self.eval_data.toNumPy()
        # Always keep default format as NumPy array if possible
        return self.eval_data

    def toDF(self):
        """
        This is a convenience function that calls the global eval method and then converts the matrix object into DataFrame.
        """
        if isinstance(self.eval_data, DataFrame):
            return self.eval_data
        if isinstance(self.eval_data, py4j.java_gateway.JavaObject):
            self.eval_data = _java2py(SparkContext._active_spark_context, self.eval_data)
        if isinstance(self.eval_data, Matrix):
            self.eval_data = self.eval_data.toDF()
            return self.eval_data
        self.eval_data = matrix.sparkSession.createDataFrame(self.toPandas())
        return self.eval_data

    def save(self, file, format='csv'):
        """
        Allows user to save a matrix to filesystem
    
        Parameters
        ----------
        file: filepath
        format: can be csv, text or binary or mm
        """
        tmp = construct_intermediate_node([self], ['save(', self.ID , ',\"', file, '\", format=\"', format, '\")\n'])
        construct_intermediate_node([tmp], [OUTPUT_ID, ' = full(0, rows=1, cols=1)\n']).eval()
    
    def _mark_as_visited(self):
        self.visited = True
        # for cleanup
        matrix.visited = matrix.visited + [ self ]
        return self

    def _register_as_input(self, execute):
        # TODO: Remove this when automatic registration of frame is resolved
        matrix.dml = [ self.ID,  ' = load(\" \", format=\"csv\")\n'] + matrix.dml
        if isinstance(self.eval_data, SUPPORTED_TYPES) and execute:
            matrix.script.input(self.ID, convertToMatrixBlock(matrix.sc, self.eval_data))
        elif execute:
            matrix.script.input(self.ID, self.toDF())
        return self

    def _register_as_output(self, execute):
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
        self._mark_as_visited()
        if self.eval_data is not None:
            self._register_as_input(execute)
        elif self.op is not None:
            # Traverse the AST
            for m in self.op.inputs:
                m._visit(execute=execute)
            self.op._visit(execute=execute)
        else:
            raise Exception('Expected either op or data to be set')
        if self.eval_data is None and self.output:
            self._register_as_output(execute)
        return self

    def print_ast(self):
        """
        Please use m.print_ast() and/or  type `m` for debugging. Here is a sample session:
        
        >>> npm = np.ones((3,3))
        >>> m1 = sml.matrix(npm + 3)
        >>> m2 = sml.matrix(npm + 5)
        >>> m3 = m1 + m2
        >>> m3
        mVar2 = load(" ", format="csv")
        mVar1 = load(" ", format="csv")
        mVar3 = mVar1 + mVar2
        save(mVar3, " ")
        >>> m3.print_ast()
        - [mVar3] (op).
          - [mVar1] (data).
          - [mVar2] (data).
        """
        return self._print_ast(0)
    
    def _print_ast(self, numSpaces):
        head = ''.join([ ' ' ]*numSpaces + [ '- [', self.ID, '] ' ])
        if self.eval_data is not None:
            out = head + '(data).\n'
        elif self.op is not None:
            ret = [ head, '(op).\n' ]
            for m in self.op.inputs:
                ret = ret + [ m._print_ast(numSpaces + 2) ]
            out = ''.join(ret)
        else:
            raise ValueError('Either op or data needs to be set')
        if numSpaces == 0:
            print(out)
        else:
            return out

    def __repr__(self):
        """
        This function helps to debug matrix class and also examine the generated PyDML script
        """
        if self.eval_data is None:
            print('# This matrix (' + self.ID + ') is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPy() or toDF() or toPandas() methods.\n' + eval([self], execute=False))
        else:
            print('# This matrix (' + self.ID + ') is backed by ' + str(type(self.eval_data)) + '. To fetch the DataFrame or NumPy array, invoke toDF() or toNumPy() method respectively.')
        return ''
    
    ######################### NumPy related methods ######################################
    
    __array_priority__ = 10.2
    ndim = 2
    
    THROW_ARRAY_CONVERSION_ERROR = False
    
    def __array__(self, dtype=np.double):
        """
        As per NumPy from Python,
        This method is called to obtain an ndarray object when needed. You should always guarantee this returns an actual ndarray object.
        
        Using this method, you get back a ndarray object, and subsequent operations on the returned ndarray object will be singlenode.
        """
        if not isinstance(self.eval_data, SUPPORTED_TYPES):
            # Only warn if there is an unevaluated operation (which could potentially generate large matrix or if data is non-supported singlenode formats)
            import inspect
            frame,filename,line_number,function_name,lines,index = inspect.stack()[1]
            msg = 'Conversion from SystemML matrix to NumPy array (occurs in ' + str(filename) + ':' + str(line_number) + ' ' + function_name + ")"
            if matrix.THROW_ARRAY_CONVERSION_ERROR:
                raise Exception('[ERROR]:' + msg)
            else:
                print('[WARN]:' + msg)
        return np.array(self.toNumPy(), dtype)
    
    def astype(self, t):
        # TODO: Throw error if incorrect type
        return self
    
    def asfptype(self):
        return self
        
    def set_shape(self,shape):
        raise NotImplementedError('Reshaping is not implemented')
    
    def get_shape(self):
        if self._shape is None:
            lhsStr, inputs = _matricize(self, [])
            rlen_ID = _get_new_var_id()
            clen_ID = _get_new_var_id()
            multiline_dml = [rlen_ID, ' = ', lhsStr, '.shape(0)\n']
            multiline_dml = multiline_dml + [clen_ID, ' = ', lhsStr, '.shape(1)\n']
            multiline_dml = multiline_dml + [OUTPUT_ID, ' = full(0, rows=2, cols=1)\n']
            multiline_dml = multiline_dml + [ OUTPUT_ID, '[0,0] = ', rlen_ID, '\n' ]
            multiline_dml = multiline_dml + [ OUTPUT_ID, '[1,0] = ', clen_ID, '\n' ]
            ret = construct_intermediate_node(inputs, multiline_dml).toNumPy()
            self._shape = tuple(np.array(ret, dtype=int).flatten())
        return self._shape 
    
    shape = property(fget=get_shape, fset=set_shape)
    
    def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
        """
        This function enables systemml matrix to be compatible with NumPy's ufuncs.
        
        Parameters
        ----------
        func:  ufunc object that was called.
        method: string indicating which Ufunc method was called (one of "__call__", "reduce", "reduceat", "accumulate", "outer", "inner").
        pos: index of self in inputs.
        inputs:  tuple of the input arguments to the ufunc
        kwargs: dictionary containing the optional input arguments of the ufunc.
        """
        if method != '__call__' or kwargs:
            return NotImplemented
        if func in matrix._numpy_to_systeml_mapping:
            fn = matrix._numpy_to_systeml_mapping[func]
        else:
            return NotImplemented
        if len(inputs) == 2:
            return fn(inputs[0], inputs[1])
        elif  len(inputs) == 1:
            return fn(inputs[0])
        else:
            raise ValueError('Unsupported number of inputs')

    def hstack(self, other):
        """
        Stack matrices horizontally (column wise). Invokes cbind internally.
        """
        return binaryMatrixFunction(self, other, 'cbind')
    
    def vstack(self, other):
        """
        Stack matrices vertically (row wise). Invokes rbind internally.
        """
        return binaryMatrixFunction(self, other, 'rbind')
            
    ######################### Arithmetic operators ######################################

    def negative(self):
        lhsStr, inputs = _matricize(self, [])
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = -', lhsStr, '\n'])
                
    def remainder(self, other):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        rhsStr, inputs = _matricize(other, inputs)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = floor(', lhsStr, '/', rhsStr, ') * ', rhsStr, '\n'])
    
    def ldexp(self, other):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        rhsStr, inputs = _matricize(other, inputs)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = ', lhsStr, '* (2**', rhsStr, ')\n'])
        
    def mod(self, other):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        rhsStr, inputs = _matricize(other, inputs)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = ', lhsStr, ' - floor(', lhsStr, '/', rhsStr, ') * ', rhsStr, '\n'])
    
    def logaddexp(self, other):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        rhsStr, inputs = _matricize(other, inputs)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = log(exp(', lhsStr, ') + exp(', rhsStr, '))\n'])
    
    def logaddexp2(self, other):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        rhsStr, inputs = _matricize(other, inputs)
        opStr =  _log_base('2**' + lhsStr + '2**' + rhsStr, 2)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = ', opStr, '\n'])

    def log1p(self):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = log(1 + ', lhsStr, ')\n'])
        
    def exp(self):
        return unaryMatrixFunction(self, 'exp')

    def exp2(self):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = 2**', lhsStr, '\n'])
    
    def square(self):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = ', lhsStr, '**2\n'])    
    
    def reciprocal(self):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = 1/', lhsStr, '\n'])
        
    def expm1(self):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = exp(', lhsStr, ') - 1\n'])
    
    def ones_like(self):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        rlen = lhsStr + '.shape(axis=0)'
        clen = lhsStr + '.shape(axis=1)'
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = full(1, rows=', rlen, ', cols=', clen, ')\n'])
    
    def zeros_like(self):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        rlen = lhsStr + '.shape(axis=0)'
        clen = lhsStr + '.shape(axis=1)'
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = full(0, rows=', rlen, ', cols=', clen, ')\n'])    
    
    def log2(self):
        return self.log(2)
    
    def log10(self):
        return self.log(10)
        
    def log(self, y=None):
        if y is None:
            return unaryMatrixFunction(self, 'log')
        else:
            return binaryMatrixFunction(self, y, 'log')

    def abs(self):
        return unaryMatrixFunction(self, 'abs')

    def sqrt(self):
        return unaryMatrixFunction(self, 'sqrt')

    def round(self):
        return unaryMatrixFunction(self, 'round')

    def floor(self):
        return unaryMatrixFunction(self, 'floor')

    def ceil(self):
        return unaryMatrixFunction(self, 'ceil')

    def sin(self):
        return unaryMatrixFunction(self, 'sin')

    def cos(self):
        return unaryMatrixFunction(self, 'cos')

    def tan(self):
        return unaryMatrixFunction(self, 'tan')

    def arcsin(self):
        return self.asin()

    def arccos(self):
        return self.acos()

    def arctan(self):
        return self.atan()
    
    def asin(self):
        return unaryMatrixFunction(self, 'asin')

    def acos(self):
        return unaryMatrixFunction(self, 'acos')

    def atan(self):
        return unaryMatrixFunction(self, 'atan')

    def rad2deg(self):
        """
        Convert angles from radians to degrees.
        """
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        # 180/pi = 57.2957795131
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = ', lhsStr, '*57.2957795131\n'])
    
    def deg2rad(self):
        """
        Convert angles from degrees to radians.
        """
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        # pi/180 = 0.01745329251
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = ', lhsStr, '*0.01745329251\n'])    
    
    def sign(self):
        return unaryMatrixFunction(self, 'sign')    

    def __add__(self, other):
        return binary_op(self, other, ' + ')

    def __sub__(self, other):
        return binary_op(self, other, ' - ')

    def __mul__(self, other):
        return binary_op(self, other, ' * ')

    def __floordiv__(self, other):
        return binary_op(self, other, ' // ')

    def __div__(self, other):
        """
        Performs division (Python 2 way).
        """
        return binary_op(self, other, ' / ')

    def __truediv__(self, other):
        """
        Performs division (Python 3 way).
        """
        return binary_op(self, other, ' / ')

    def __mod__(self, other):
        return binary_op(self, other, ' % ')

    def __pow__(self, other):
        return binary_op(self, other, ' ** ')

    def __radd__(self, other):
        return binary_op(other, self, ' + ')

    def __rsub__(self, other):
        return binary_op(other, self, ' - ')

    def __rmul__(self, other):
        return binary_op(other, self, ' * ')

    def __rfloordiv__(self, other):
        return binary_op(other, self, ' // ')

    def __rdiv__(self, other):
        return binary_op(other, self, ' / ')

    def __rtruediv__(self, other):
        """
        Performs division (Python 3 way).
        """
        return binary_op(other, self, ' / ')

    def __rmod__(self, other):
        return binary_op(other, self, ' % ')

    def __rpow__(self, other):
        return binary_op(other, self, ' ** ')

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
        return binary_op(self, other, ' < ')

    def __le__(self, other):
        return binary_op(self, other, ' <= ')

    def __gt__(self, other):
        return binary_op(self, other, ' > ')

    def __ge__(self, other):
        return binary_op(self, other,' >= ')

    def __eq__(self, other):
        return binary_op(self, other, ' == ')

    def __ne__(self, other):
        return binary_op(self, other, ' != ')
    
    # TODO: Cast the output back into scalar and return boolean results
    def __and__(self, other):
        return binary_op(other, self, ' & ')

    def __or__(self, other):
        return binary_op(other, self, ' | ')

    def logical_not(self):
        inputs = []
        lhsStr, inputs = _matricize(self, inputs)
        return construct_intermediate_node(inputs, [OUTPUT_ID, ' = !', lhsStr, '\n'])
    
    def remove_empty(self, axis=None):
        """
        Removes all empty rows or columns from the input matrix target X according to specified axis.
        
        Parameters
        ----------
        axis : int (0 or 1)
        """
        if axis is None:
            raise ValueError('axis is a mandatory argument for remove_empty')
        if axis == 0:
            return self._parameterized_helper_fn(self, 'removeEmpty',  { 'target':self, 'margin':'rows' })
        elif axis == 1:
            return self._parameterized_helper_fn(self, 'removeEmpty',  { 'target':self, 'margin':'cols' })
        else:
            raise ValueError('axis for remove_empty needs to be either 0 or 1.')
    
    def replace(self, pattern=None, replacement=None):
        """
        Removes all empty rows or columns from the input matrix target X according to specified axis.
        
        Parameters
        ----------
        pattern : float or int
        replacement : float or int
        """
        if pattern is None or not isinstance(pattern, (float, int)):
            raise ValueError('pattern should be of type float or int')
        if replacement is None or not isinstance(replacement, (float, int)):
            raise ValueError('replacement should be of type float or int')
        return self._parameterized_helper_fn(self, 'replace',  { 'target':self, 'pattern':pattern, 'replacement':replacement })
    
    def _parameterized_helper_fn(self, fnName, **kwargs):
        """
        Helper to invoke parameterized builtin function
        """
        dml_script = ''
        lhsStr, inputs = _matricize(self, [])
        dml_script = [OUTPUT_ID, ' = ', fnName, '(', lhsStr ]
        first_arg = True
        for key in kwargs:
            if first_arg:
                first_arg = False
            else:
                dml_script = dml_script + [ ', ' ]
            v = kwargs[key]
            if isinstance(v, str):
                dml_script = dml_script + [key, '=\"', v, '\"' ]
            elif isinstance(v, matrix):
                dml_script = dml_script + [key, '=', v.ID]
            else:
                dml_script = dml_script + [key, '=', str(v) ]
        dml_script = dml_script + [ ')\n' ]
        return construct_intermediate_node(inputs, dml_script)
            
    ######################### Aggregation functions ######################################

    def prod(self):
        """
        Return the product of all cells in matrix
        """
        return self._aggFn('prod', None)
        
    def sum(self, axis=None):
        """
        Compute the sum along the specified axis
        
        Parameters
        ----------
        axis : int, optional
        """
        return self._aggFn('sum', axis)

    def mean(self, axis=None):
        """
        Compute the arithmetic mean along the specified axis
        
        Parameters
        ----------
        axis : int, optional
        """
        return self._aggFn('mean', axis)

    def var(self, axis=None):
        """
        Compute the variance along the specified axis.
        We assume that delta degree of freedom is 1 (unlike NumPy which assumes ddof=0).
        
        Parameters
        ----------
        axis : int, optional
        """
        return self._aggFn('var', axis)
        
    def moment(self, moment=1, axis=None):
        """
        Calculates the nth moment about the mean
        
        Parameters
        ----------
        moment : int
            can be 1, 2, 3 or 4
        axis : int, optional
        """
        if moment == 1:
            return self.mean(axis)
        elif moment == 2:
            return self.var(axis)
        elif moment == 3 or moment == 4:
            return self._moment_helper(moment, axis)
        else:
            raise ValueError('The specified moment is not supported:' + str(moment))
        
    def _moment_helper(self, k, axis=0):
        dml_script = ''
        lhsStr, inputs = _matricize(self, [])
        dml_script = [OUTPUT_ID, ' = moment(', lhsStr, ', ', str(k), ')\n' ]
        dml_script = [OUTPUT_ID, ' = moment(', lhsStr, ', ', str(k), ')\n' ]
        if axis is None:
            dml_script = [OUTPUT_ID, ' = moment(full(', lhsStr, ', rows=length(', lhsStr, '), cols=1), ', str(k), ')\n' ]
        elif axis == 0:
            dml_script = [OUTPUT_ID, ' = full(0, rows=nrow(', lhsStr, '), cols=1)\n' ]
            dml_script = dml_script + [ 'parfor(i in 1:nrow(', lhsStr, '), check=0):\n' ]
            dml_script = dml_script + [ '\t', OUTPUT_ID, '[i-1, 0] = moment(full(', lhsStr, '[i-1,], rows=ncol(', lhsStr, '), cols=1), ', str(k), ')\n\n' ]
        elif axis == 1:
            dml_script = [OUTPUT_ID, ' = full(0, rows=1, cols=ncol(', lhsStr, '))\n' ]
            dml_script = dml_script + [ 'parfor(i in 1:ncol(', lhsStr, '), check=0):\n' ]
            dml_script = dml_script + [ '\t', OUTPUT_ID, '[0, i-1] = moment(', lhsStr, '[,i-1], ', str(k), ')\n\n' ]
        else:
            raise ValueError('Incorrect axis:' + axis)
        return construct_intermediate_node(inputs, dml_script)
        
    def sd(self, axis=None):
        """
        Compute the standard deviation along the specified axis
        
        Parameters
        ----------
        axis : int, optional
        """
        return self._aggFn('sd', axis)

    def max(self, other=None, axis=None):
        """
        Compute the maximum value along the specified axis
        
        Parameters
        ----------
        other: matrix or numpy array (& other supported types) or scalar
        axis : int, optional
        """
        if other is not None and axis is not None:
            raise ValueError('Both axis and other cannot be not None')
        elif other is None and axis is not None:
            return self._aggFn('max', axis)
        else:
            return binaryMatrixFunction(self, other, 'max')

    def min(self, other=None, axis=None):
        """
        Compute the minimum value along the specified axis
        
        Parameters
        ----------
        other: matrix or numpy array (& other supported types) or scalar
        axis : int, optional
        """
        if other is not None and axis is not None:
            raise ValueError('Both axis and other cannot be not None')
        elif other is None and axis is not None:
            return self._aggFn('min', axis)
        else:
            return binaryMatrixFunction(self, other, 'min')

    def argmin(self, axis=None):
        """
        Returns the indices of the minimum values along an axis.
        
        Parameters
        ----------
        axis : int, optional  (only axis=1, i.e. rowIndexMax is supported in this version)
        """
        return self._aggFn('argmin', axis)

    def argmax(self, axis=None):
        """
        Returns the indices of the maximum values along an axis.
        
        Parameters
        ----------
        axis : int, optional (only axis=1, i.e. rowIndexMax is supported in this version)
        """
        return self._aggFn('argmax', axis)

    def cumsum(self, axis=None):
        """
        Returns the indices of the maximum values along an axis.
        
        Parameters
        ----------
        axis : int, optional (only axis=0, i.e. cumsum along the rows is supported in this version)
        """
        return self._aggFn('cumsum', axis)

    def transpose(self):
        """
        Transposes the matrix.
        """
        return self._aggFn('transpose', None)

    def trace(self):
        """
        Return the sum of the cells of the main diagonal square matrix
        """
        return self._aggFn('trace', None)

    def _aggFn(self, fnName, axis):
        """
        Common function that is called for functions that have axis as parameter.
        """
        dml_script = ''
        lhsStr, inputs = _matricize(self, [])
        if axis is None:
            dml_script = [OUTPUT_ID, ' = ', fnName, '(', lhsStr, ')\n']
        else:
            dml_script = [OUTPUT_ID, ' = ', fnName, '(', lhsStr, ', axis=', str(axis) ,')\n']
        return construct_intermediate_node(inputs, dml_script)

    ######################### Indexing operators ######################################

    def __getitem__(self, index):
        """
        Implements evaluation of right indexing operations such as m[1,1], m[0:1,], m[:, 0:1]
        """
        return construct_intermediate_node([self], [OUTPUT_ID, ' = ', self.ID ] + getIndexingDML(index) + [ '\n' ])

    # Performs deep copy if the matrix is backed by data
    def _prepareForInPlaceUpdate(self):
        temp = matrix(self.eval_data, op=self.op)
        for op in self.referenced:
            op.inputs = [temp if x.ID==self.ID else x for x in op.inputs]
        self.ID, temp.ID = temp.ID, self.ID # Copy even the IDs as the IDs might be used to create DML
        self.op = DMLOp([temp], dml=[self.ID, " = ", temp.ID])
        self.eval_data = None
        temp.referenced = self.referenced + [ self.op ]
        self.referenced = []

    def __setitem__(self, index, value):
        """
        Implements evaluation of left indexing operations such as m[1,1]=2
        """
        self._prepareForInPlaceUpdate()
        if isinstance(value, matrix) or isinstance(value, DMLOp):
            self.op.inputs = self.op.inputs + [ value ]
        if isinstance(value, matrix):
            value.referenced = value.referenced + [ self.op ]
        self.op.dml = self.op.dml + [ '\n', self.ID ] + getIndexingDML(index) + [ ' = ',  getValue(value), '\n']

    # Not implemented: conj, hyperbolic/inverse-hyperbolic functions(i.e. sinh, arcsinh, cosh, ...), bitwise operator, xor operator, isreal, iscomplex, isfinite, isinf, isnan, copysign, nextafter, modf, frexp, trunc  
    _numpy_to_systeml_mapping = {np.add: __add__, np.subtract: __sub__, np.multiply: __mul__, np.divide: __div__, np.logaddexp: logaddexp, np.true_divide: __truediv__, np.floor_divide: __floordiv__, np.negative: negative, np.power: __pow__, np.remainder: remainder, np.mod: mod, np.fmod: __mod__, np.absolute: abs, np.rint: round, np.sign: sign, np.exp: exp, np.exp2: exp2, np.log: log, np.log2: log2, np.log10: log10, np.expm1: expm1, np.log1p: log1p, np.sqrt: sqrt, np.square: square, np.reciprocal: reciprocal, np.ones_like: ones_like, np.zeros_like: zeros_like, np.sin: sin, np.cos: cos, np.tan: tan, np.arcsin: arcsin, np.arccos: arccos, np.arctan: arctan, np.deg2rad: deg2rad, np.rad2deg: rad2deg, np.greater: __gt__, np.greater_equal: __ge__, np.less: __lt__, np.less_equal: __le__, np.not_equal: __ne__, np.equal: __eq__, np.logical_not: logical_not, np.logical_and: __and__, np.logical_or: __or__, np.maximum: max, np.minimum: min, np.signbit: sign, np.ldexp: ldexp, np.dot:dot}
