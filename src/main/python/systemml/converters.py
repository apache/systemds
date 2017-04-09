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

__all__ = [ 'getNumCols', 'convertToMatrixBlock', 'convertToNumPyArr', 'convertToPandasDF', 'SUPPORTED_TYPES' , 'convertToLabeledDF']

import numpy as np
import pandas as pd
import math
from pyspark.context import SparkContext
from scipy.sparse import coo_matrix, spmatrix, csr_matrix
from .classloader import *

SUPPORTED_TYPES = (np.ndarray, pd.DataFrame, spmatrix)

def getNumCols(numPyArr):
    if numPyArr.ndim == 1:
        return 1
    else:
        return numPyArr.shape[1]


def convertToLabeledDF(sparkSession, X, y=None):
    from pyspark.ml.feature import VectorAssembler
    if y is not None:
        pd1 = pd.DataFrame(X)
        pd2 = pd.DataFrame(y, columns=['label'])
        pdf = pd.concat([pd1, pd2], axis=1)
        inputColumns = ['C' + str(i) for i in pd1.columns]
        outputColumns = inputColumns + ['label']
    else:
        pdf = pd.DataFrame(X)
        inputColumns = ['C' + str(i) for i in pdf.columns]
        outputColumns = inputColumns
    assembler = VectorAssembler(inputCols=inputColumns, outputCol='features')
    out = assembler.transform(sparkSession.createDataFrame(pdf, outputColumns))
    if y is not None:
        return out.select('features', 'label')
    else:
        return out.select('features')

def _convertSPMatrixToMB(sc, src):
    src = coo_matrix(src,  dtype=np.float64)
    numRows = src.shape[0]
    numCols = src.shape[1]
    data = src.data
    row = src.row.astype(np.int32)
    col = src.col.astype(np.int32)
    nnz = len(src.col)
    buf1 = bytearray(data.tostring())
    buf2 = bytearray(row.tostring())
    buf3 = bytearray(col.tostring())
    createJavaObject(sc, 'dummy')
    return sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.convertSciPyCOOToMB(buf1, buf2, buf3, numRows, numCols, nnz)
            
def _convertDenseMatrixToMB(sc, src):
    numCols = getNumCols(src)
    numRows = src.shape[0]
    arr = src.ravel().astype(np.float64)
    buf = bytearray(arr.tostring())
    createJavaObject(sc, 'dummy')
    return sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.convertPy4JArrayToMB(buf, numRows, numCols)

def _copyRowBlock(i, sc, ret, src, numRowsPerBlock,  rlen, clen):
    rowIndex = int(i / numRowsPerBlock)
    tmp = src[i:min(i+numRowsPerBlock, rlen),]
    mb = _convertSPMatrixToMB(sc, tmp) if isinstance(src, spmatrix) else _convertDenseMatrixToMB(sc, tmp)
    sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.copyRowBlocks(mb, rowIndex, ret, numRowsPerBlock, rlen, clen)
    return i
    
def convertToMatrixBlock(sc, src, maxSizeBlockInMB=8):
    if not isinstance(sc, SparkContext):
        raise TypeError('sc needs to be of type SparkContext')
    isSparse = True if isinstance(src, spmatrix) else False
    src = np.asarray(src, dtype=np.float64) if not isSparse else src
    if len(src.shape) != 2:
        src_type = str(type(src).__name__)
        raise TypeError('Expected 2-dimensional ' + src_type + ', instead passed ' + str(len(src.shape)) + '-dimensional ' + src_type)
    # Ignoring sparsity for computing numRowsPerBlock for now
    numRowsPerBlock = int(math.ceil((maxSizeBlockInMB*1000000) / (src.shape[1]*8)))
    multiBlockTransfer = False if numRowsPerBlock >= src.shape[0] else True
    if not multiBlockTransfer:
        return _convertSPMatrixToMB(sc, src) if isSparse else _convertDenseMatrixToMB(sc, src)
    else:
        # Since coo_matrix does not have range indexing
        src = csr_matrix(src) if isSparse else src
        rlen = int(src.shape[0])
        clen = int(src.shape[1])
        ret = sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.allocateDenseOrSparse(rlen, clen, isSparse)
        [ _copyRowBlock(i, sc, ret, src, numRowsPerBlock,  rlen, clen) for i in range(0, src.shape[0], numRowsPerBlock) ]
        sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.postProcessAfterCopying(ret)
        return ret

def convertToNumPyArr(sc, mb):
    if isinstance(sc, SparkContext):
        numRows = mb.getNumRows()
        numCols = mb.getNumColumns()
        createJavaObject(sc, 'dummy')
        buf = sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.convertMBtoPy4JDenseArr(mb)
        return np.frombuffer(buf, count=numRows*numCols, dtype=np.float64).reshape((numRows, numCols))
    else:
        raise TypeError('sc needs to be of type SparkContext') # TODO: We can generalize this by creating py4j gateway ourselves


def convertToPandasDF(X):
    if not isinstance(X, pd.DataFrame):
        return pd.DataFrame(X, columns=['C' + str(i) for i in range(getNumCols(X))])
    return X
