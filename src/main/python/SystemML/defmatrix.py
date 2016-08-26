#!/usr/bin/python
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

import numpy as np

from . import pydml, MLContext
from .converters import *
from pyspark import SparkContext, RDD
from pyspark.sql import DataFrame, SQLContext

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
    def __init__(self, inputs, dml=None):
        self.inputs = inputs
        self.dml = dml
        
    def _visit(self, execute=True):
        matrix.dml = matrix.dml + self.dml
            

def reset():
    for m in matrix.visited:
        m.visited = False
    matrix.visited = []
        
def binaryOp(lhs, rhs, opStr):
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
    for m in outputs:
        if outputDF:
            m.data = results.getDataFrame(m.ID)
        else:
            m.data = results.getNumPyArray(m.ID)
    
# Instead of inheriting from np.matrix
class matrix(object):
    systemmlVarID = 0
    dml = []
    script = None
    ml = None
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
        eval([self], outputDF=False)
        
    def toPandas(self):
        if self.data is None:
            self.eval()
        return convertToPandasDF(self.data)    
    
    def toNumPyArray(self):
        if self.data is None:
            self.eval()
        if isinstance(self.data, DataFrame):
            self.data = self.data.toPandas().as_matrix()
        # Always keep default format as NumPy array if possible
        return self.data
    
    def toDataFrame(self):
        if self.data is None:
            self.eval(outputDF=True)
        if not isinstance(self.data, DataFrame):
            if MLResults.sqlContext is None:
                MLResults.sqlContext = SQLContext(matrix.sc)
            self.data = sqlContext.createDataFrame(self.toPandas())
        return self.data
        
    def _visit(self, execute=True):
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
                matrix.script.out(self.ID)
        return self
    
    def __repr__(self):
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
        dmlOp = DMLOp([self])
        out = matrix(None, op=dmlOp)
        if axis is None:
            dmlOp.dml = [out.ID, ' = ', fnName, '(', self.ID, ')\n']
        else:
            dmlOp.dml = [out.ID, ' = ', fnName, '(', self.ID, ', axis=', str(axis) ,')\n']
        return out        

    def dot(self, other):
        dmlOp = DMLOp([self])
        out = matrix(None, op=dmlOp)
        dmlOp.dml = [out.ID, ' = dot(', self.ID, ', ', other.ID, ')\n']
        return out       
            
__all__ = [ 'setSparkContext', 'matrix', 'eval']