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

from py4j.protocol import Py4JJavaError, Py4JError
import traceback
import os
from pyspark.sql import DataFrame, SQLContext
from pyspark.rdd import RDD


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
                for key, value in nargs.items():
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
