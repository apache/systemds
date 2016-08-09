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

from __future__ import division
from py4j.protocol import Py4JJavaError, Py4JError
import traceback
import os
from pyspark.context import SparkContext 
from pyspark.sql import DataFrame, SQLContext
from pyspark.rdd import RDD
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import metrics
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
import sys
from pyspark.ml import Estimator, Model
from scipy.sparse import spmatrix
from scipy.sparse import coo_matrix

SUPPORTED_TYPES = (np.ndarray, pd.DataFrame, spmatrix)

class MLContext(object):

    """
    Simple wrapper class for MLContext in SystemML.jar
    ...
    Attributes
    ----------
    ml : MLContext
        A reference to the java MLContext
    sc : SparkContext
        The SparkContext that has been specified during initialization
    """

    def __init__(self, sc, *args):
        """
        If initialized with a SparkContext, will connect to the Java MLContext
        class.
        args:
            sc (SparkContext): the current SparkContext
            monitor (boolean=False): Whether to monitor the performance
            forceSpark (boolean=False): Whether to force execution on spark
        returns:
            MLContext: Instance of MLContext
        """

        try:
            monitorPerformance = (args[0] if len(args) > 0 else False)
            setForcedSparkExecType = (args[1] if len(args) > 1 else False)
            self.sc = sc
            self.ml = sc._jvm.org.apache.sysml.api.MLContext(sc._jsc, monitorPerformance, setForcedSparkExecType)
            self.sqlCtx = SQLContext(sc)
        except Py4JError:
            traceback.print_exc()

    def reset(self):
        """
        Call this method of you want to clear any RDDs set via
        registerInput or registerOutput
        """
        try:
            self.ml.reset()
        except Py4JJavaError:
            traceback.print_exc()

    def execute(self, dmlScriptFilePath, *args):
        """
        Executes the script in spark-mode by passing the arguments to the
        MLContext java class.
        Returns:
            MLOutput: an instance of the MLOutput-class
        """
        numArgs = len(args) + 1
        try:
            if numArgs == 1:
                jmlOut = self.ml.execute(dmlScriptFilePath)
                mlOut = MLOutput(jmlOut, self.sc)
                return mlOut
            elif numArgs == 2:
                jmlOut = self.ml.execute(dmlScriptFilePath, args[0])
                mlOut = MLOutput(jmlOut, self.sc)
                return mlOut
            elif numArgs == 3:
                jmlOut = self.ml.execute(dmlScriptFilePath, args[0], args[1])
                mlOut = MLOutput(jmlOut, self.sc)
                return mlOut
            elif numArgs == 4:
                jmlOut = self.ml.execute(dmlScriptFilePath, args[0], args[1], args[2])
                mlOut = MLOutput(jmlOut, self.sc)
                return mlOut
            else:
                raise TypeError('Arguments do not match MLContext-API')
        except Py4JJavaError:
            traceback.print_exc()

    def executeScript(self, dmlScript, nargs=None, outputs=None, isPyDML=False, configFilePath=None):
        """
        Executes the script in spark-mode by passing the arguments to the
        MLContext java class.
        Returns:
            MLOutput: an instance of the MLOutput-class
        """
        try:
            # Register inputs as needed
            if nargs is not None:
                for key, value in list(nargs.items()):
                    if isinstance(value, DataFrame):
                        self.registerInput(key, value)
                        del nargs[key]
                    else:
                        nargs[key] = str(value)
            else:
                nargs = {}

            # Register outputs as needed
            if outputs is not None:
                for out in outputs:
                    self.registerOutput(out)

            # Execute script
            jml_out = self.ml.executeScript(dmlScript, nargs, isPyDML, configFilePath)
            ml_out = MLOutput(jml_out, self.sc)
            return ml_out
        except Py4JJavaError:
            traceback.print_exc()

    def registerInput(self, varName, src, *args):
        """
        Method to register inputs used by the DML script.
        Supported format:
        1. DataFrame
        2. CSV/Text (as JavaRDD<String> or JavaPairRDD<LongWritable, Text>)
        3. Binary blocked RDD (JavaPairRDD<MatrixIndexes,MatrixBlock>))
        Also overloaded to support metadata information such as format, rlen, clen, ...
        Please note the variable names given below in quotes correspond to the variables in DML script.
        These variables need to have corresponding read/write associated in DML script.
        Currently, only matrix variables are supported through registerInput/registerOutput interface.
        To pass scalar variables, use named/positional arguments (described later) or wrap them into matrix variable.
        """
        numArgs = len(args) + 2

        if hasattr(src, '_jdf'):
            rdd = src._jdf
        elif hasattr(src, '_jrdd'):
            rdd = src._jrdd
        else:
            rdd = src

        try:
            if numArgs == 2:
                self.ml.registerInput(varName, rdd)
            elif numArgs == 3:
                self.ml.registerInput(varName, rdd, args[0])
            elif numArgs == 4:
                self.ml.registerInput(varName, rdd, args[0], args[1])
            elif numArgs == 5:
                self.ml.registerInput(varName, rdd, args[0], args[1], args[2])
            elif numArgs == 6:
                self.ml.registerInput(varName, rdd, args[0], args[1], args[2], args[3])
            elif numArgs == 7:
                self.ml.registerInput(varName, rdd, args[0], args[1], args[2], args[3], args[4])
            elif numArgs == 10:
                self.ml.registerInput(varName, rdd, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])
            else:
                raise TypeError('Arguments do not match MLContext-API')
        except Py4JJavaError:
            traceback.print_exc()

    def registerOutput(self, varName):
        """
        Register output variables used in the DML script
        args:
            varName: (String) The name used in the DML script
        """

        try:
            self.ml.registerOutput(varName)
        except Py4JJavaError:
            traceback.print_exc()

    def getDmlJson(self):
        try:
            return self.ml.getMonitoringUtil().getRuntimeInfoInJSONFormat()
        except Py4JJavaError:
            traceback.print_exc()


class MLOutput(object):

    """
    This is a simple wrapper object that returns the output of execute from MLContext
    ...
    Attributes
    ----------
    jmlOut MLContext:
        A reference to the MLOutput object through py4j
    """

    def __init__(self, jmlOut, sc):
        self.jmlOut = jmlOut
        self.sc = sc

    def getBinaryBlockedRDD(self, varName):
        raise Exception('Not supported in Python MLContext')
        #try:
        #    rdd = RDD(self.jmlOut.getBinaryBlockedRDD(varName), self.sc)
        #    return rdd
        #except Py4JJavaError:
        #    traceback.print_exc()

    def getMatrixCharacteristics(self, varName):
        raise Exception('Not supported in Python MLContext')
        #try:
        #    chars = self.jmlOut.getMatrixCharacteristics(varName)
        #    return chars
        #except Py4JJavaError:
        #    traceback.print_exc()

    def getDF(self, sqlContext, varName):
        try:
            jdf = self.jmlOut.getDF(sqlContext._ssql_ctx, varName)
            df = DataFrame(jdf, sqlContext)
            return df
        except Py4JJavaError:
            traceback.print_exc()

    def getPandasDF(self, sqlContext, varName):
        df = self.toDF(sqlContext, varName).sort('ID').drop('ID')
        return df.toPandas()
        
    def getMLMatrix(self, sqlContext, varName):
        raise Exception('Not supported in Python MLContext')
        #try:
        #    mlm = self.jmlOut.getMLMatrix(sqlContext._scala_SQLContext, varName)
        #    return mlm
        #except Py4JJavaError:
        #    traceback.print_exc()

    def getStringRDD(self, varName, format):
        raise Exception('Not supported in Python MLContext')
        #try:
        #    rdd = RDD(self.jmlOut.getStringRDD(varName, format), self.sc)
        #    return rdd
        #except Py4JJavaError:
        #    traceback.print_exc()

def getNumCols(numPyArr):
    if numPyArr.ndim == 1:
        return 1
    else:
        return numPyArr.shape[1]
       
def convertToMatrixBlock(sc, src):
    if isinstance(src, spmatrix):
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
        return sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.convertSciPyCOOToMB(buf1, buf2, buf3, numRows, numCols, nnz)
    elif isinstance(sc, SparkContext):
        src = np.asarray(src)
        numCols = getNumCols(src)
        numRows = src.shape[0]
        arr = src.ravel().astype(np.float64)
        buf = bytearray(arr.tostring())
        return sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.convertPy4JArrayToMB(buf, numRows, numCols)
    else:
        raise TypeError('sc needs to be of type SparkContext') # TODO: We can generalize this by creating py4j gateway ourselves
    

def convertToNumpyArr(sc, mb):
    if isinstance(sc, SparkContext):
        numRows = mb.getNumRows()
        numCols = mb.getNumColumns()
        buf = sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.convertMBtoPy4JDenseArr(mb)
        return np.frombuffer(buf, count=numRows*numCols, dtype=np.float64)
    else:
        raise TypeError('sc needs to be of type SparkContext') # TODO: We can generalize this by creating py4j gateway ourselves

def convertToPandasDF(X):
    if not isinstance(X, pd.DataFrame):
        return pd.DataFrame(X, columns=['C' + str(i) for i in range(getNumCols(X))])
    return X
            
def tolist(inputCols):
    return list(inputCols)

def assemble(sqlCtx, pdf, inputCols, outputCol):
    tmpDF = sqlCtx.createDataFrame(pdf, tolist(pdf.columns))
    assembler = VectorAssembler(inputCols=tolist(inputCols), outputCol=outputCol)
    return assembler.transform(tmpDF)

class mllearn:
    class BaseSystemMLEstimator(Estimator):
    # TODO: Allow users to set featuresCol (with default 'features') and labelCol (with default 'label')
    
        # Returns a model after calling fit(df) on Estimator object on JVM    
        def _fit(self, X):
            if hasattr(X, '_jdf') and 'features' in X.columns and 'label' in X.columns:
                self.model = self.estimator.fit(X._jdf)
                return self
            else:
                raise Exception('Incorrect usage: Expected dataframe as input with features/label as columns')
        
        # Returns a model after calling fit(X:MatrixBlock, y:MatrixBlock) on Estimator object on JVM  
        def fit(self, X, y=None, params=None):
            if y is None:
                return self._fit(X)
            elif y is not None and isinstance(X, SUPPORTED_TYPES):
                if self.transferUsingDF:
                    pdfX = convertToPandasDF(X)
                    pdfY = convertToPandasDF(y)
                    if getNumCols(pdfY) != 1:
                        raise Exception('y should be a column vector')
                    if pdfX.shape[0] != pdfY.shape[0]:
                        raise Exception('Number of rows of X and y should match')
                    colNames = pdfX.columns
                    pdfX['label'] = pdfY[pdfY.columns[0]]
                    df = assemble(self.sqlCtx, pdfX, colNames, 'features').select('features', 'label')
                    self.model = self.estimator.fit(df._jdf)
                else:
                    numColsy = getNumCols(y)
                    if numColsy != 1:
                        raise Exception('Expected y to be a column vector')
                    self.model = self.estimator.fit(convertToMatrixBlock(self.sc, X), convertToMatrixBlock(self.sc, y))
                if self.setOutputRawPredictionsToFalse:
                    self.model.setOutputRawPredictions(False)
                return self
            else:
                raise Exception('Unsupported input type')
        
        def transform(self, X):
            return self.predict(X)
        
        # Returns either a DataFrame or MatrixBlock after calling transform(X:MatrixBlock, y:MatrixBlock) on Model object on JVM    
        def predict(self, X):
            if isinstance(X, SUPPORTED_TYPES):
                if self.transferUsingDF:
                    pdfX = convertToPandasDF(X)
                    df = assemble(self.sqlCtx, pdfX, pdfX.columns, 'features').select('features')
                    retjDF = self.model.transform(df._jdf)
                    retDF = DataFrame(retjDF, self.sqlCtx)
                    retPDF = retDF.sort('ID').select('prediction').toPandas()
                    if isinstance(X, np.ndarray):
                        return retPDF.as_matrix().flatten()
                    else:
                        return retPDF
                else:
                    retNumPy = convertToNumpyArr(self.sc, self.model.transform(convertToMatrixBlock(self.sc, X)))
                    if isinstance(X, np.ndarray):
                        return retNumPy
                    else:
                        return retNumPy # TODO: Convert to Pandas
            elif hasattr(X, '_jdf'):
                if 'features' in X.columns:
                    # No need to assemble as input DF is likely coming via MLPipeline
                    df = X
                else:
                    assembler = VectorAssembler(inputCols=X.columns, outputCol='features')
                    df = assembler.transform(X)
                retjDF = self.model.transform(df._jdf)
                retDF = DataFrame(retjDF, self.sqlCtx)
                # Return DF
                return retDF.sort('ID')
            else:
                raise Exception('Unsupported input type')
                
    class BaseSystemMLClassifier(BaseSystemMLEstimator):

        # Scores the predicted value with ground truth 'y'
        def score(self, X, y):
            return metrics.accuracy_score(y, self.predict(X))    
    
    class BaseSystemMLRegressor(BaseSystemMLEstimator):

        # Scores the predicted value with ground truth 'y'
        def score(self, X, y):
            return metrics.r2_score(y, self.predict(X), multioutput='variance_weighted')

    
    # Or we can create new Python project with package structure
    class LogisticRegression(BaseSystemMLClassifier):

        # See https://apache.github.io/incubator-systemml/algorithms-reference for usage
        def __init__(self, sqlCtx, penalty='l2', fit_intercept=True, max_iter=100, max_inner_iter=0, tol=0.000001, C=1.0, solver='newton-cg', transferUsingDF=False):
            self.sqlCtx = sqlCtx
            self.sc = sqlCtx._sc
            self.uid = "logReg"
            self.estimator = self.sc._jvm.org.apache.sysml.api.ml.LogisticRegression(self.uid, self.sc._jsc.sc())
            self.estimator.setMaxOuterIter(max_iter)
            self.estimator.setMaxInnerIter(max_inner_iter)
            if C <= 0:
                raise Exception('C has to be positive')
            reg = 1.0 / C
            self.estimator.setRegParam(reg)
            self.estimator.setTol(tol)
            self.estimator.setIcpt(int(fit_intercept))
            self.transferUsingDF = transferUsingDF
            self.setOutputRawPredictionsToFalse = True
            if penalty != 'l2':
                raise Exception('Only l2 penalty is supported')
            if solver != 'newton-cg':
                raise Exception('Only newton-cg solver supported')

    class LinearRegression(BaseSystemMLRegressor):

        # See https://apache.github.io/incubator-systemml/algorithms-reference for usage
        def __init__(self, sqlCtx, fit_intercept=True, max_iter=100, tol=0.000001, C=1.0, solver='newton-cg', transferUsingDF=False):
            self.sqlCtx = sqlCtx
            self.sc = sqlCtx._sc
            self.uid = "lr"
            if solver == 'newton-cg' or solver == 'direct-solve':
                self.estimator = self.sc._jvm.org.apache.sysml.api.ml.LinearRegression(self.uid, self.sc._jsc.sc(), solver)
            else:
                raise Exception('Only newton-cg solver supported')
            self.estimator.setMaxIter(max_iter)
            if C <= 0:
                raise Exception('C has to be positive')
            reg = 1.0 / C
            self.estimator.setRegParam(reg)
            self.estimator.setTol(tol)
            self.estimator.setIcpt(int(fit_intercept))
            self.transferUsingDF = transferUsingDF
            self.setOutputRawPredictionsToFalse = False


    class SVM(BaseSystemMLClassifier):

        # See https://apache.github.io/incubator-systemml/algorithms-reference for usage
        def __init__(self, sqlCtx, fit_intercept=True, max_iter=100, tol=0.000001, C=1.0, is_multi_class=False, transferUsingDF=False):
            self.sqlCtx = sqlCtx
            self.sc = sqlCtx._sc
            self.uid = "svm"
            self.estimator = self.sc._jvm.org.apache.sysml.api.ml.SVM(self.uid, self.sc._jsc.sc(), is_multi_class)
            self.estimator.setMaxIter(max_iter)
            if C <= 0:
                raise Exception('C has to be positive')
            reg = 1.0 / C
            self.estimator.setRegParam(reg)
            self.estimator.setTol(tol)
            self.estimator.setIcpt(int(fit_intercept))
            self.transferUsingDF = transferUsingDF
            self.setOutputRawPredictionsToFalse = False    

    class NaiveBayes(BaseSystemMLClassifier):

        # See https://apache.github.io/incubator-systemml/algorithms-reference for usage
        def __init__(self, sqlCtx, laplace=1.0, transferUsingDF=False):
            self.sqlCtx = sqlCtx
            self.sc = sqlCtx._sc
            self.uid = "nb"
            self.estimator = self.sc._jvm.org.apache.sysml.api.ml.NaiveBayes(self.uid, self.sc._jsc.sc())
            self.estimator.setLaplace(laplace)
            self.transferUsingDF = transferUsingDF
            self.setOutputRawPredictionsToFalse = False