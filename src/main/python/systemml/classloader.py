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

__all__ = ['createJavaObject']

import os

try:
    import py4j.java_gateway
    from py4j.java_gateway import JavaObject
    from pyspark import SparkContext
except ImportError:
    raise ImportError('Unable to import `pyspark`. Hint: Make sure you are running with PySpark.')

def _createJavaObject(sc, obj_type):
    if obj_type == 'mlcontext':
        return sc._jvm.org.apache.sysml.api.mlcontext.MLContext(sc._jsc)
    elif obj_type == 'dummy':
        return sc._jvm.org.apache.sysml.utils.SystemMLLoaderUtils()
    else:
        raise ValueError('Incorrect usage: supported values: mlcontext or dummy')

def _getJarFileName(sc, suffix):
    import imp, fnmatch
    jar_file_name = '_ignore.jar'
    java_dir = os.path.join(imp.find_module("systemml")[1], "systemml-java")
    for file in os.listdir(java_dir):
        if fnmatch.fnmatch(file, 'systemml-*-SNAPSHOT' + suffix + '.jar') or fnmatch.fnmatch(file, 'systemml-*' + suffix + '.jar'):
            jar_file_name = os.path.join(java_dir, file)
    return jar_file_name

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
        # First load SystemML
        jar_file_name = _getJarFileName(sc, '')
        x = _getLoaderInstance(sc, jar_file_name, 'org.apache.sysml.utils.SystemMLLoaderUtils', hint + 'SystemML.jar')
        x.loadSystemML(jar_file_name)
        try:
            ret = _createJavaObject(sc, obj_type)
        except (py4j.protocol.Py4JError, TypeError):
            raise ImportError(err_msg + ' Hint: ' + hint + jar_file_name)
        # Now load caffe2dml
        jar_file_name = _getJarFileName(sc, '-extra')
        x = _getLoaderInstance(sc, jar_file_name, 'org.apache.sysml.api.dl.Caffe2DMLLoader', hint + 'systemml-*-extra.jar')
        x.loadCaffe2DML(jar_file_name)
        return ret