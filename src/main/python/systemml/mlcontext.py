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

__all__ = ['MLResults', 'MLContext', 'Script', 'dml', 'pydml']

import os

try:
    import py4j.java_gateway
    from py4j.java_gateway import JavaObject
except ImportError:
    raise ImportError('Unable to import JavaObject from py4j.java_gateway. Hint: Make sure you are running with pyspark')

from pyspark import SparkContext
import pyspark.mllib.common

from .converters import *

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
        self.sc = sc

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
        df = _java2py(self.sc, jdf)
        return df

    def toNumPy(self):
        """
        Convert the Matrix to a NumPy Array.

        Returns
        -------
        NumPy Array
            A NumPy Array representing the Matrix object.
        """
        np_array = convertToNumPyArr(self.sc, self._java_matrix.toBinaryBlockMatrix().getMatrixBlock())
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
        self.sc = sc

    def __repr__(self):
        return "MLResults"

    def get(self, *outputs):
        """
        Parameters
        ----------
        outputs: string, list of strings
            Output variables as defined inside the DML script.
        """
        outs = [_java2py(self.sc, self._java_results.get(out)) for out in outputs]
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
    """
    def __init__(self, scriptString, scriptType="dml"):
        self.scriptString = scriptString
        self.scriptType = scriptType
        self._input = {}
        self._output = []

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
            self._input[args[0]] = args[1]
        for name, value in kwargs.items():
            self._input[name] = value
        return self

    def output(self, *names):
        """
        Parameters
        ----------
        names: string, list of strings
            Output variables as defined inside the DML script.
        """
        self._output.extend(names)
        return self


class MLContext(object):
    """
    Wrapper around the new SystemML MLContext.

    Parameters
    ----------
    sc: SparkContext
        SparkContext
    """
    def __init__(self, sc):
        if not isinstance(sc, SparkContext):
            raise ValueError("Expected sc to be a SparkContext, got " % sc)
        self._sc = sc
        self._ml = sc._jvm.org.apache.sysml.api.mlcontext.MLContext(sc._jsc)

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
        if script.scriptType == "dml":
            if scriptString.endswith(".dml"):
                if os.path.exists(scriptString):
                    script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile(scriptString)
                else:
                    raise ValueError("path: %s does not exist" % scriptString)
            else:
                script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dml(scriptString)
        elif script.scriptType == "pydml":
            if scriptString.endswith(".pydml"):
                if os.path.exists(scriptString):
                    script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromFile(scriptString)
                else:
                    raise ValueError("path: %s does not exist" % scriptString)
            else:
                script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydml(scriptString)

        for key, val in script._input.items():
            # `in` is a reserved word ("keyword") in Python, so `script_java.in(...)` is not
            # allowed. Therefore, we use the following code in which we retrieve a function
            # representing `script_java.in`, and then call it with the arguments.  This is in
            # lieu of adding a new `input` method on the JVM side, as that would complicate use
            # from Scala/Java.
            py4j.java_gateway.get_method(script_java, "in")(key, _py2java(self._sc, val))
        for val in script._output:
            script_java.out(val)
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
