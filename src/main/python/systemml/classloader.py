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

__all__ = ['createJavaObject', 'jvm_stdout', 'default_jvm_stdout', 'default_jvm_stdout_parallel_flush', 'set_default_jvm_stdout', 'get_spark_context' ]

import os
import numpy as np
import pandas as pd
import threading, time

try:
    import py4j.java_gateway
    from py4j.java_gateway import JavaObject
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
except ImportError:
    raise ImportError('Unable to import `pyspark`. Hint: Make sure you are running with PySpark.')

_loadedSystemML = False
def get_spark_context():
    """
    Internal method to get already initialized SparkContext.  Developers should always use
    get_spark_context() instead of SparkContext._active_spark_context to ensure SystemML loaded.

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
        
_in_jvm_stdout = False
default_jvm_stdout = True
default_jvm_stdout_parallel_flush = True
def set_default_jvm_stdout(enable, parallel_flush=True):
    """
    This is useful utility method to get the output of the driver JVM from within a Jupyter notebook

    Parameters
    ----------
    enable: boolean
        Should flush the stdout by default when mlcontext.execute is invoked
        
    parallel_flush: boolean
        Should flush the stdout in parallel
    """
    global default_jvm_stdout, default_jvm_stdout_parallel_flush
    default_jvm_stdout = enable
    default_jvm_stdout_parallel_flush = parallel_flush
    
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
        self.util = get_spark_context()._jvm.org.apache.sysml.api.ml.Utils()
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
        global _in_jvm_stdout
        if _in_jvm_stdout:
            # Allow for nested jvm_stdout    
            self.donotRedirect = True
        else:
            self.donotRedirect = False
            self.util.startRedirectStdOut()
            if self.parallel_flush:
                self.t.start()
            _in_jvm_stdout = True

    def __exit__(self, *args):
        global _in_jvm_stdout
        if not self.donotRedirect:
            if self.parallel_flush:
                self.stop = True
                self.t.join()
            print(self.util.stopRedirectStdOut())
            _in_jvm_stdout = False


_initializedSparkSession = False
def _createJavaObject(sc, obj_type):
    # -----------------------------------------------------------------------------------
    # Avoids race condition between locking of metastore_db of Scala SparkSession and PySpark SparkSession.
    # This is done at toDF() rather than import level to avoid creation of SparkSession in worker processes.
    global _initializedSparkSession
    if not _initializedSparkSession:
        _initializedSparkSession = True
        SparkSession.builder.getOrCreate().createDataFrame(pd.DataFrame(np.array([[1,2],[3,4]])))
    # -----------------------------------------------------------------------------------
    if obj_type == 'mlcontext':
        return sc._jvm.org.apache.sysml.api.mlcontext.MLContext(sc._jsc)
    elif obj_type == 'dummy':
        return sc._jvm.org.apache.sysml.utils.SystemMLLoaderUtils()
    else:
        raise ValueError('Incorrect usage: supported values: mlcontext or dummy')

def _getJarFileNames(sc):
    import imp, fnmatch
    jar_file_name = '_ignore.jar'
    java_dir = os.path.join(imp.find_module("systemml")[1], "systemml-java")
    jar_file_names = []
    for file in os.listdir(java_dir):
        if fnmatch.fnmatch(file, 'systemml-*-SNAPSHOT.jar') or fnmatch.fnmatch(file, 'systemml-*.jar'):
            jar_file_names = jar_file_names + [ os.path.join(java_dir, file) ]
    return jar_file_names

def _getLoaderInstance(sc, jar_file_name, className, hint):
    err_msg = 'Unable to load systemml-*.jar into current pyspark session.'
    if os.path.isfile(jar_file_name):
        sc._jsc.addJar(jar_file_name)
        jar_file_url = sc._jvm.java.io.File(jar_file_name).toURI().toURL()
        url_class = sc._jvm.java.net.URL
        jar_file_url_arr = sc._gateway.new_array(url_class, 1)
        jar_file_url_arr[0] = jar_file_url
        url_class_loader = sc._jvm.java.net.URLClassLoader(jar_file_url_arr, sc._jsc.getClass().getClassLoader())
        c1 = sc._jvm.java.lang.Class.forName(className, True, url_class_loader)
        return c1.newInstance()
    else:
        raise ImportError(err_msg + ' Hint: Download the jar from http://systemml.apache.org/download and ' + hint )

def createJavaObject(sc, obj_type):
    """
    Performs appropriate check if SystemML.jar is available and returns the handle to MLContext object on JVM
    
    Parameters
    ----------
    sc: SparkContext
        SparkContext
    obj_type: Type of object to create ('mlcontext' or 'dummy')
    """
    try:
        return _createJavaObject(sc, obj_type)
    except (py4j.protocol.Py4JError, TypeError):
        ret = None
        err_msg = 'Unable to load systemml-*.jar into current pyspark session.'
        hint = 'Provide the following argument to pyspark: --driver-class-path '
        jar_file_names = _getJarFileNames(sc)
        if len(jar_file_names) != 2:
            raise ImportError('Expected only systemml and systemml-extra jars, but found ' + str(jar_file_names))
        for jar_file_name in jar_file_names:
            if 'extra' in jar_file_name:
                x = _getLoaderInstance(sc, jar_file_name, 'org.apache.sysml.api.dl.Caffe2DMLLoader', hint + 'systemml-*-extra.jar')
                x.loadCaffe2DML(jar_file_name)
            else:
                x = _getLoaderInstance(sc, jar_file_name, 'org.apache.sysml.utils.SystemMLLoaderUtils', hint + 'systemml-*.jar')
                x.loadSystemML(jar_file_name)
        try:
            ret = _createJavaObject(sc, obj_type)
        except (py4j.protocol.Py4JError, TypeError):
            raise ImportError(err_msg + ' Hint: ' + hint + jar_file_name)
        return ret
