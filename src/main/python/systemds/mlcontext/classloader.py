
__all__ = ['createJavaObject']

import os
import pandas as pd

try:
    import py4j.java_gateway
    from pyspark.sql import SparkSession
except ImportError:
    raise ImportError('Unable  to import `pyspark`. Hint: Make sure you are running with PySpark.')


def _createJavaObject(sc, obj_type):

    if obj_type == 'mlcontext':
        return sc._jvm.org.apache.sysds.api.mlcontext.MLContext(sc._jsc)
    else:
        raise ValueError('Incorrect usage: supported values: mlcontext')

def _getJarFileName(sc, suffix):
    import imp, fnmatch
    jar_file_name = '_ignore.jar'
    java_dir = os.path.join(imp.find_module("systemds")[1], "systemds-java")
    java_dir = os.path.join(java_dir, "lib")
    for file in os.listdir(java_dir):
        if fnmatch.fnmatch(file, 'systemds-*-SNAPSHOT' + suffix + '.jar'):
            jar_file_name = os.path.join(java_dir, file)
    return jar_file_name

def _getLoaderInstance(sc, jar_file_name, className, hint):
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
        raise ImportError('Hint: Download the jar from http://systemds.apache.org/download')

def createJavaObject(sc, obj_type):
    """
    Performs appropriate check if SystemDS.jar is available and
    returns the handle to MLContext object on JVM

    :param sc: SparkContext
    :param obj_type: Type of object to create ('mlcontext')
    :return:
    """
    try:
        return _createJavaObject(sc, obj_type)
    except (TypeError):
        ret = None
        hint = 'Provide the following argument to pyspark: --driver-class-path '
        # First Load SystemDS
        jar_file_name = _getJarFileName(sc, '')
        x = _getLoaderInstance(sc, jar_file_name, 'org.apache.sysds.utils.SystemDSLoaderUtils', hint + 'SystemDS.jar')
        x.loadSystemDS(jar_file_name)
        try:
            ret = _createJavaObject(sc, obj_type)
        except (py4j.protocol.Py4JError, TypeError):
            raise ImportError('Hint: ' + hint + jar_file_name)
        return ret
