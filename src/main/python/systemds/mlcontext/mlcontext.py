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
from py4j.protocol import Py4JNetworkError

from systemds.utils.helpers import get_module_dir
from pyspark import SparkContext

# class MLContext(object):
#     """
#     Create an MLContext object with spark session or Spark Context
#     """
#
#     def __init__(self):
#         """
#         Starts a new instance of MLContext
#         """
#
#         systemds_java_path = os.path.join(get_module_dir(), "systemds-java")
#         # nt means its Windows
#         cp_separator = ";" if os.name == "nt" else ":"
#         lib_cp = os.path.join(systemds_java_path, "lib", "*")
#         systemds_cp = os.path.join(systemds_java_path, "*")
#         classpath = cp_separator.join([lib_cp, systemds_cp])
#
#         command = ["java", "-cp", classpath]
#
#         sys_root = os.environ.get("SYSTEMDS_ROOT")
#         if sys_root != None:
#             files = glob(os.path.join(sys_root, "conf", "log4j*.properties"))
#             if len(files) > 1:
#                 print("WARNING: Multiple logging files")
#             if len(files) == 0:
#                 print("WARNING: No log4j file found at: "
#                       + os.path.join(sys_root, "conf")
#                       + " therefore using default settings")
#             else:
#                 # print("Using Log4J file at " + files[0])
#                 command.append("-Dlog4j.configuration=file:" + files[0])
#         else:
#             print("Default Log4J used, since environment $SYSTEMDS_ROOT not set")
#
#         command.append("org.apache.sysds.api.MLContext")
#
#         port = self.__get_open_port()
#         command.append(str(port))
#
#         process = Popen(command, stdout=PIPE, stdin=PIPE, stderr=PIPE)
#
#         # Py4j connect to the started process.
#         gateway_parameters = GatewayParameters(
#             port=port, eager_load=True, read_timeout=5)
#         self.java_gateway = JavaGateway(
#             gateway_parameters=gateway_parameters, java_process=process)
#
#     def close(self):
#         """Close the connection to the java process"""
#         process : Popen = self.java_gateway.java_process
#         self.java_gateway.shutdown()
#         os.kill(process.pid, 14)
#
#     def __get_open_port(self):
#         """Get a random available port."""
#         # TODO Verify that it is not taking some critical ports change to select a good port range.
#         # TODO If it tries to select a port already in use, find anotherone. (recursion)
#         # https://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python
#         import socket
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(("", 0))
#         s.listen(1)
#         port = s.getsockname()[1]
#         s.close()
#         return port


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
        elif script.scriptType == "pydml":
            if scriptString.endswith(".pydml"):
                if scriptString.startswith("http"):
                    script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromUrl(scriptString)
                elif os.path.exists(scriptString):
                    script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromFile(scriptString)
                elif script.isResource == True:
                    script_java = self._sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromResource(
                        scriptString)
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
            if isinstance(val, py4j.java_gateway.JavaObject):
                py4j.java_gateway.get_method(script_java, "in")(key, val)
            else:
                py4j.java_gateway.get_method(script_java, "in")(key, _py2java(self._sc, val))
        for val in script._output:
            script_java.out(val)
        return MLResults(self._ml.execute(script_java), self._sc)