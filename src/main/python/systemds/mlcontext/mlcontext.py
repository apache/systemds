__all__ = ["MLContext"]

import copy
import os
import time
from glob import glob
from queue import Empty, Queue
from subprocess import PIPE, Popen
from threading import Lock, Thread
from time import sleep
from typing import Dict, Iterable, Sequence, Tuple, Union

import numpy as np
from py4j.java_gateway import GatewayParameters, JavaGateway
from py4j.java_gateway import JavaObject
from py4j.protocol import Py4JNetworkError

import pyspark.mllib.common

from systemds.utils.helpers import get_module_dir
from pyspark import SparkContext
from pyspark.sql import SparkSession

from .classloader import *

class MLContext(object):
    """
    Wrapper around the SystemDS MLContext.

    Parameters
    ----------
    sc: SparkContext
        SparkContext
    """

    def __init__(self, sc):
        if not isinstance(sc, SparkContext):
            raise ValueError("Expected sc to be a SparkContext, got " % sc)
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
        if script.scriptType == "dml":
            if scriptString.endswith(".dml"):
                if scriptString.startswith("http"):
                    script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromUrl(scriptString)
                elif os.path.exists(scriptString):
                    script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile(scriptString)
                elif script.isResource == True:
                    script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromResource(
                        scriptString)
                else:
                    raise ValueError("path: %s does not exist" % scriptString)
            else:
                script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dml(scriptString)

        for key, val in script._input.items():
            # `in` is a reserved word ("keyword") in Python, so `script_java.in(...)` is not
            # allowed. Therefore, we use the following code in which we retrieve a function
            # representing `script_java.in`, and then call it with the arguments.  This is in
            # lieu of adding a new `input` method on the JVM side, as that would complicate use
            # from Scala/Java.
            if isinstance(val, py4j.java_gateway.JavaObject):
                py4j.java_gateway.get_method(script_java, "in")(key, val)
            else:
                py4j.java_gateway.get_method(script_java, "in")(key, _py2java(self._sc, val))
        for val in script._output:
            script_java.out(val)
        return MLResults(self._ml.execute(script_java), self._sc)
