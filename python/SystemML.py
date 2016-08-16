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
import os

from py4j.java_gateway import JavaObject
from py4j.java_collections import ListConverter, JavaArray, JavaList
from pyspark import SparkContext, RDD
from pyspark.mllib.common import _java2py, _py2java
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
from pyspark.sql import DataFrame


class MLResults(object):
    """
    Wrapper around the Java ML Results object.

    Parameters
    ----------
    results: JavaObject
        A Java MLResults object as returned by calling ml.execute()

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
    path: string
        Can be either a file path to a DML script or a DML script itself.
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
            where name is a string and currently supported value formats
            are double, string, rdds and list of such object.

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

    def out(self, *names):
        """
        Parameters
        ----------
        outputs: string, list of strings
            Output variables as defined inside the DML script.
        """
        self._output.extend(names)
        return self


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
            script_java.input(key, _py2java(self._sc, val))
        for val in script._output:
            script_java.out(val)
        return MLResults(self._ml.execute(script_java), self._sc)
