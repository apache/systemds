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

# Methods to create Script object
script_factory_methods = [ 'dml', 'pydml', 'dmlFromResource', 'pydmlFromResource', 'dmlFromFile', 'pydmlFromFile', 'dmlFromUrl', 'pydmlFromUrl' ]
# Utility methods
util_methods = [ 'jvm_stdout', '_java2py',  'getHopDAG' ]
__all__ = ['MLResults', 'MLContext', 'Script', 'Matrix' ] + script_factory_methods + util_methods

import os
import numpy as np
import pandas as pd
import threading, time
    
try:
    import py4j.java_gateway
    from py4j.java_gateway import JavaObject
    from pyspark import SparkContext
    from pyspark.conf import SparkConf
    import pyspark.mllib.common
    from pyspark.sql import SparkSession
except ImportError:
    raise ImportError('Unable to import `pyspark`. Hint: Make sure you are running with PySpark.')

from .converters import *
from .classloader import *

_loadedSystemML = False
def _get_spark_context():
    """
    Internal method to get already initialized SparkContext.  Developers should always use
    _get_spark_context() instead of SparkContext._active_spark_context to ensure SystemML loaded.

    Returns
    -------
    sc: SparkContext
        SparkContext
    """
    if SparkContext._active_spark_context is not None:
        sc = SparkContext._active_spark_context
        global _loadedSystemML
        if not _loadedSystemML:
            createJavaObject(sc, 'dummy')
            _loadedSystemML = True
        return sc
    else:
        raise Exception('Expected spark context to be created.')

# This is useful utility class to get the output of the driver JVM from within a Jupyter notebook
# Example usage:
# with jvm_stdout():
#    ml.execute(script)
class jvm_stdout(object):
    """
    This is useful utility class to get the output of the driver JVM from within a Jupyter notebook

    Parameters
    ----------
    parallel_flush: boolean
        Should flush the stdout in parallel
    """
    def __init__(self, parallel_flush=False):
        self.util = _get_spark_context()._jvm.org.apache.sysml.api.ml.Utils()
        self.parallel_flush = parallel_flush
        self.t = threading.Thread(target=self.flush_stdout)
        self.stop = False
        
    def flush_stdout(self):
        while not self.stop: 
            time.sleep(1) # flush stdout every 1 second
            str = self.util.flushStdOut()
            if str != '':
                str = str[:-1] if str.endswith('\n') else str
                print(str)
    
    def __enter__(self):
        self.util.startRedirectStdOut()
        if self.parallel_flush:
            self.t.start()

    def __exit__(self, *args):
        if self.parallel_flush:
            self.stop = True
            self.t.join()
        print(self.util.stopRedirectStdOut())
        

def getHopDAG(ml, script, lines=None, conf=None, apply_rewrites=True, with_subgraph=False):
    """
    Compile a DML / PyDML script.

    Parameters
    ----------
    ml: MLContext instance
        MLContext instance.
        
    script: Script instance
        Script instance defined with the appropriate input and output variables.
    
    lines: list of integers
        Optional: only display the hops that have begin and end line number equals to the given integers.
    
    conf: SparkConf instance
        Optional spark configuration
        
    apply_rewrites: boolean
        If True, perform static rewrites, perform intra-/inter-procedural analysis to propagate size information into functions and apply dynamic rewrites
    
    with_subgraph: boolean
        If False, the dot graph will be created without subgraphs for statement blocks. 
    
    Returns
    -------
    hopDAG: string
        hop DAG in dot format 
    """
    if not isinstance(script, Script):
        raise ValueError("Expected script to be an instance of Script")
    scriptString = script.scriptString
    script_java = script.script_java
    lines = [ int(x) for x in lines ] if lines is not None else [int(-1)]
    sc = _get_spark_context()
    if conf is not None:
        hopDAG = sc._jvm.org.apache.sysml.api.mlcontext.MLContextUtil.getHopDAG(ml._ml, script_java, lines, conf._jconf, apply_rewrites, with_subgraph)
    else:
        hopDAG = sc._jvm.org.apache.sysml.api.mlcontext.MLContextUtil.getHopDAG(ml._ml, script_java, lines, apply_rewrites, with_subgraph)
    return hopDAG

def dml(scriptString):
    """
    Create a dml script object based on a string.

    Parameters
    ----------
    scriptString: string
        Can be a path to a dml script or a dml script itself.

    Returns
    -------
    script: Script instance
        Instance of a script object.
    """
    if not isinstance(scriptString, str):
        raise ValueError("scriptString should be a string, got %s" % type(scriptString))
    return Script(scriptString, scriptType="dml")

def dmlFromResource(resourcePath):
    """
    Create a dml script object based on a resource path.

    Parameters
    ----------
    resourcePath: string
        Path to a dml script on the classpath.

    Returns
    -------
    script: Script instance
        Instance of a script object.
    """
    if not isinstance(resourcePath, str):
        raise ValueError("resourcePath should be a string, got %s" % type(resourcePath))
    return Script(resourcePath, scriptType="dml", isResource=True)


def pydml(scriptString):
    """
    Create a pydml script object based on a string.

    Parameters
    ----------
    scriptString: string
        Can be a path to a pydml script or a pydml script itself.

    Returns
    -------
    script: Script instance
        Instance of a script object.
    """
    if not isinstance(scriptString, str):
        raise ValueError("scriptString should be a string, got %s" % type(scriptString))
    return Script(scriptString, scriptType="pydml")

def pydmlFromResource(resourcePath):
    """
    Create a pydml script object based on a resource path.

    Parameters
    ----------
    resourcePath: string
        Path to a pydml script on the classpath.

    Returns
    -------
    script: Script instance
        Instance of a script object.
    """
    if not isinstance(resourcePath, str):
        raise ValueError("resourcePath should be a string, got %s" % type(resourcePath))
    return Script(resourcePath, scriptType="pydml", isResource=True)

def dmlFromFile(filePath):
    """
    Create a dml script object based on a file path.

    Parameters
    ----------
    filePath: string
        Path to a dml script.

    Returns
    -------
    script: Script instance
        Instance of a script object.
    """
    if not isinstance(filePath, str):
        raise ValueError("filePath should be a string, got %s" % type(filePath))
    return Script(filePath, scriptType="dml", isResource=False, scriptFormat="file")
    
def pydmlFromFile(filePath):
    """
    Create a pydml script object based on a file path.

    Parameters
    ----------
    filePath: string
        Path to a pydml script.

    Returns
    -------
    script: Script instance
        Instance of a script object.
    """
    if not isinstance(filePath, str):
        raise ValueError("filePath should be a string, got %s" % type(filePath))
    return Script(filePath, scriptType="pydml", isResource=False, scriptFormat="file")
    

def dmlFromUrl(url):
    """
    Create a dml script object based on a url.

    Parameters
    ----------
    url: string
        URL to a dml script.

    Returns
    -------
    script: Script instance
        Instance of a script object.
    """
    if not isinstance(url, str):
        raise ValueError("url should be a string, got %s" % type(url))
    return Script(url, scriptType="dml", isResource=False, scriptFormat="url")

def pydmlFromUrl(url):
    """
    Create a pydml script object based on a url.

    Parameters
    ----------
    url: string
        URL to a pydml script.

    Returns
    -------
    script: Script instance
        Instance of a script object.
    """
    if not isinstance(url, str):
        raise ValueError("url should be a string, got %s" % type(url))
    return Script(url, scriptType="pydml", isResource=False, scriptFormat="url")

def _java2py(sc, obj):
    """ Convert Java object to Python. """
    # TODO: Port this private PySpark function.
    obj = pyspark.mllib.common._java2py(sc, obj)
    if isinstance(obj, JavaObject):
        class_name = obj.getClass().getSimpleName()
        if class_name == 'Matrix':
            obj = Matrix(obj, sc)
    return obj


def _py2java(sc, obj):
    """ Convert Python object to Java. """
    if isinstance(obj, SUPPORTED_TYPES):
        obj = convertToMatrixBlock(sc, obj)
    else:
        if isinstance(obj, Matrix):
            obj = obj._java_matrix
        # TODO: Port this private PySpark function.
        obj = pyspark.mllib.common._py2java(sc, obj)
    return obj


class Matrix(object):
    """
    Wrapper around a Java Matrix object.

    Parameters
    ----------
    javaMatrix: JavaObject
        A Java Matrix object as returned by calling `ml.execute().get()`.

    sc: SparkContext
        SparkContext
    """
    def __init__(self, javaMatrix, sc):
        self._java_matrix = javaMatrix
        self._sc = sc

    def __repr__(self):
        return "Matrix"

    def toDF(self):
        """
        Convert the Matrix to a PySpark SQL DataFrame.

        Returns
        -------
        PySpark SQL DataFrame
            A PySpark SQL DataFrame representing the matrix, with
            one "__INDEX" column containing the row index (since Spark
            DataFrames are unordered), followed by columns of doubles
            for each column in the matrix.
        """
        jdf = self._java_matrix.toDF()
        df = _java2py(self._sc, jdf)
        return df

    def toNumPy(self):
        """
        Convert the Matrix to a NumPy Array.

        Returns
        -------
        NumPy Array
            A NumPy Array representing the Matrix object.
        """
        np_array = convertToNumPyArr(self._sc, self._java_matrix.toMatrixBlock())
        return np_array


class MLResults(object):
    """
    Wrapper around a Java ML Results object.

    Parameters
    ----------
    results: JavaObject
        A Java MLResults object as returned by calling `ml.execute()`.

    sc: SparkContext
        SparkContext
    """
    def __init__(self, results, sc):
        self._java_results = results
        self._sc = sc

    def __repr__(self):
        return "MLResults"

    def get(self, *outputs):
        """
        Parameters
        ----------
        outputs: string, list of strings
            Output variables as defined inside the DML script.
        """
        outs = [_java2py(self._sc, self._java_results.get(out)) for out in outputs]
        if len(outs) == 1:
            return outs[0]
        return outs


class Script(object):
    """
    Instance of a DML/PyDML Script.

    Parameters
    ----------
    scriptString: string
        Can be either a file path to a DML script or a DML script itself.

    scriptType: string
        Script language, either "dml" for DML (R-like) or "pydml" for PyDML (Python-like).

    isResource: boolean
        If true, scriptString is a path to a resource on the classpath
    
    scriptFormat: string
        Optional script format, either "auto" or "url" or "file" or "resource" or "string"
    """
    def __init__(self, scriptString, scriptType="dml", isResource=False, scriptFormat="auto"):
        self.sc = _get_spark_context()
        self.scriptString = scriptString
        self.scriptType = scriptType
        self.isResource = isResource
        if scriptFormat != "auto":
            if scriptFormat == "url" and self.scriptType == "dml":
                self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromUrl(scriptString)
            elif scriptFormat == "url" and self.scriptType == "pydml":
                self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromUrl(scriptString)
            elif scriptFormat == "file" and self.scriptType == "dml":
                self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile(scriptString)
            elif scriptFormat == "file" and self.scriptType == "pydml":
                self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromFile(scriptString)
            elif isResource and self.scriptType == "dml":
                self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromResource(scriptString)
            elif isResource and self.scriptType == "pydml":
                self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromResource(scriptString)
            elif scriptFormat == "string" and self.scriptType == "dml":
                self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dml(scriptString)
            elif scriptFormat == "string" and self.scriptType == "pydml":
                self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydml(scriptString)
            else:
                raise ValueError('Unsupported script format' + scriptFormat)
        elif self.scriptType == "dml":
            if scriptString.endswith(".dml"):
                if scriptString.startswith("http"):
                    self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromUrl(scriptString)
                elif os.path.exists(scriptString):
                    self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile(scriptString)
                elif self.isResource == True:
                    self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromResource(scriptString)
                else:
                    raise ValueError("path: %s does not exist" % scriptString)
            else:
                self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dml(scriptString)
        elif self.scriptType == "pydml":
            if scriptString.endswith(".pydml"):
                if scriptString.startswith("http"):
                    self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromUrl(scriptString)
                elif os.path.exists(scriptString):
                    self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromFile(scriptString)
                elif self.isResource == True:
                    self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromResource(scriptString)
                else:
                    raise ValueError("path: %s does not exist" % scriptString)
            else:
                self.script_java = self.sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydml(scriptString)

    
    def getScriptString(self):
        """
        Obtain the script string (in unicode).
        """
        return self.script_java.getScriptString()
    
    def setScriptString(self, scriptString):
        """
        Set the script string.
        
        Parameters
        ----------
        scriptString: string
            Can be either a file path to a DML script or a DML script itself.
        """
        self.scriptString = scriptString
        self.script_java.setScriptString(scriptString)
        return self

    def getInputVariables(self):
        """
        Obtain the input variable names.
        """
        return self.script_java.getInputVariables()

    def getOutputVariables(self):
        """
        Obtain the output variable names.
        """
        return self.script_java.getOutputVariables()

    def clearIOS(self):
        """
        Clear the inputs, outputs, and symbol table.
        """
        self.script_java.clearIOS()
        return self
    
    def clearIO(self):
        """
        Clear the inputs and outputs, but not the symbol table.
        """
        self.script_java.clearIO()
        return self
    
    def clearAll(self):
        """
        Clear the script string, inputs, outputs, and symbol table.
        """
        self.script_java.clearAll()
        return self
    
    def clearInputs(self):
        """
        Clear the inputs.
        """
        self.script_java.clearInputs()
        return self
    
    def clearOutputs(self):
        """
        Clear the outputs.
        """
        self.script_java.clearOutputs()
        return self
    
    def clearSymbolTable(self):
        """
        Clear the symbol table.
        """
        self.script_java.clearSymbolTable()
        return self
        
    def results(self):
        """
        Obtain the results of the script execution.
        """
        return MLResults(self.script_java.results(), self.sc)
    
    def getResults(self):
        """
        Obtain the results of the script execution.
        """
        return MLResults(self.script_java.getResults(), self.sc)
        
    def setResults(self, results):
        """
        Set the results of the script execution.
        """
        self.script_java.setResults(results._java_results)
        return self
        
    def isDML(self):
        """
        Is the script type DML?
        """
        return self.script_java.isDML()
    
    def isPYDML(self):
        """
        Is the script type DML?
        """
        return self.script_java.isPYDML()
    
    def getScriptExecutionString(self):
        """
        Generate the script execution string, which adds read/load/write/save
        statements to the beginning and end of the script to execute.
        """
        return self.script_java.getScriptExecutionString()    
    
    def __repr__(self):
        return "Script"

    def info(self):
        """
        Display information about the script as a String. This consists of the
        script type, inputs, outputs, input parameters, input variables, output
        variables, the symbol table, the script string, and the script execution string.
        """
        return self.script_java.info()

    def displayInputs(self):
        """
        Display the script inputs.
        """
        return self.script_java.displayInputs()
    
    def displayOutputs(self):
        """
        Display the script outputs.
        """
        return self.script_java.displayOutputs()
        
    def displayInputParameters(self):
        """
        Display the script input parameters.
        """
        return self.script_java.displayInputParameters()
    
    def displayInputVariables(self):
        """
        Display the script input variables.
        """
        return self.script_java.displayInputVariables()
        
    def displayOutputVariables(self):
        """
        Display the script output variables.
        """
        return self.script_java.displayOutputVariables()
        
    def displaySymbolTable(self):
        """
        Display the script symbol table.
        """
        return self.script_java.displaySymbolTable()
        
    def getName(self):
        """
        Obtain the script name.
        """
        return self.script_java.getName()
        
    def setName(self, name):
        """
        Set the script name.
        """
        self.script_java.setName(name)
        return self
        
    def getScriptType(self):
        """
        Obtain the script type.
        """
        return self.scriptType
        
    def input(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: name, value tuple
            where name is a string, and currently supported value formats
            are double, string, dataframe, rdd, and list of such object.

        kwargs: dict of name, value pairs
            To know what formats are supported for name and value, look above.
        """
        if args and len(args) != 2:
            raise ValueError("Expected name, value pair.")
        elif args:
            self._setInput(args[0], args[1])
        for name, value in kwargs.items():
            self._setInput(name, value)
        return self

    def _setInput(self, key, val):
        # `in` is a reserved word ("keyword") in Python, so `script_java.in(...)` is not
        # allowed. Therefore, we use the following code in which we retrieve a function
        # representing `script_java.in`, and then call it with the arguments.  This is in
        # lieu of adding a new `input` method on the JVM side, as that would complicate use
        # from Scala/Java.
        if isinstance(val, py4j.java_gateway.JavaObject):
            py4j.java_gateway.get_method(self.script_java, "in")(key, val)
        else:
            py4j.java_gateway.get_method(self.script_java, "in")(key, _py2java(self.sc, val))
    
    
    def output(self, *names):
        """
        Parameters
        ----------
        names: string, list of strings
            Output variables as defined inside the DML script.
        """
        for val in names:
            self.script_java.out(val)
        return self


class MLContext(object):
    """
    Wrapper around the new SystemML MLContext.

    Parameters
    ----------
    sc: SparkContext or SparkSession
        An instance of pyspark.SparkContext or pyspark.sql.SparkSession.
    """
    def __init__(self, sc):
        if isinstance(sc, pyspark.sql.session.SparkSession):
            sc = sc._sc
        elif not isinstance(sc, SparkContext):
            raise ValueError("Expected sc to be a SparkContext or SparkSession, got " % str(type(sc)))
        self._sc = sc
        self._ml = createJavaObject(sc, 'mlcontext')

    def __repr__(self):
        return "MLContext"
        
    def execute(self, script):
        """
        Execute a DML / PyDML script.

        Parameters
        ----------
        script: Script instance
            Script instance defined with the appropriate input and output variables.

        Returns
        -------
        ml_results: MLResults
            MLResults instance.
        """
        if not isinstance(script, Script):
            raise ValueError("Expected script to be an instance of Script")
        scriptString = script.scriptString
        script_java = script.script_java
        return MLResults(self._ml.execute(script_java), self._sc)

    def setStatistics(self, statistics):
        """
        Whether or not to output statistics (such as execution time, elapsed time)
        about script executions.

        Parameters
        ----------
        statistics: boolean
        """
        self._ml.setStatistics(bool(statistics))
        return self

    def setGPU(self, enable):
        """
        Whether or not to enable GPU.

        Parameters
        ----------
        enable: boolean
        """
        self._ml.setGPU(bool(enable))
        return self
    
    def setForceGPU(self, enable):
        """
        Whether or not to force the usage of GPU operators.

        Parameters
        ----------
        enable: boolean
        """
        self._ml.setForceGPU(bool(enable))
        return self
        
    def setStatisticsMaxHeavyHitters(self, maxHeavyHitters):
        """
        The maximum number of heavy hitters that are printed as part of the statistics.

        Parameters
        ----------
        maxHeavyHitters: int
        """
        self._ml.setStatisticsMaxHeavyHitters(maxHeavyHitters)
        return self

    def setExplain(self, explain):
        """
        Explanation about the program. Mainly intended for developers.

        Parameters
        ----------
        explain: boolean
        """
        self._ml.setExplain(bool(explain))
        return self

    def setExplainLevel(self, explainLevel):
        """
        Set explain level.

        Parameters
        ----------
        explainLevel: string
            Can be one of "hops", "runtime", "recompile_hops", "recompile_runtime"
            or in the above in upper case.
        """
        self._ml.setExplainLevel(explainLevel)
        return self

    def setConfigProperty(self, propertyName, propertyValue):
        """
        Set configuration property, such as setConfigProperty("localtmpdir", "/tmp/systemml").

        Parameters
        ----------
        propertyName: String
        propertyValue: String
        """
        self._ml.setConfigProperty(propertyName, propertyValue)
        return self

    def setConfig(self, configFilePath):
        """
        Set SystemML configuration based on a configuration file.

        Parameters
        ----------
        configFilePath: String
        """
        self._ml.setConfig(configFilePath)
        return self

    def resetConfig(self):
        """
        Reset configuration settings to default values.
        """
        self._ml.resetConfig()
        return self

    def version(self):
        """Display the project version."""
        return self._ml.version()

    def buildTime(self):
        """Display the project build time."""
        return self._ml.buildTime()

    def info(self):
        """Display the project information."""
        return self._ml.info().toString()

    def isExplain(self):
        """Returns True if program instruction details should be output, False otherwise."""
        return self._ml.isExplain()

    def isStatistics(self):
        """Returns True if program execution statistics should be output, False otherwise."""
        return self._ml.isStatistics()

    def isGPU(self):
        """Returns True if GPU mode is enabled, False otherwise."""
        return self._ml.isGPU()

    def isForceGPU(self):
        """Returns True if "force" GPU mode is enabled, False otherwise."""
        return self._ml.isForceGPU()

    def close(self):
        """
        Closes this MLContext instance to cleanup buffer pool, static/local state and scratch space.
        Note the SparkContext is not explicitly closed to allow external reuse.
        """
        self._ml.close()
        return self
