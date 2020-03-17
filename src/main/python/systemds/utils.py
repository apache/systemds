# ------------------------------------------------------------------------------
#  Copyright 2020 Graz University of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ------------------------------------------------------------------------------

import os
import platform
import subprocess
from fnmatch import fnmatch
from itertools import chain
from typing import List, Iterable, Dict
from importlib.util import find_spec

from py4j.java_gateway import JavaGateway

__all__ = ['get_gateway', 'create_params_string']

JAVA_GATEWAY = None
MODULE_NAME = 'systemds'
# TODO remove standalone
JAR_FILE_STANDALONE = 'systemds-*-standalone.jar'
JAR_FILE_NAMES = ['systemds-*-SNAPSHOT.jar', 'systemds-*.jar', JAR_FILE_STANDALONE]
AUTO_START_JMLC_SERVER = True


def set_auto_start_jmlc(should_start_server: bool) -> None:
    """
    Can activate/deactivate automatic start of JMLC API Gateway.
    Has to be called before any communication took place (`.compute()` was called).

    :param should_start_server: activate/deactivate
    """
    global AUTO_START_JMLC_SERVER
    AUTO_START_JMLC_SERVER = should_start_server


def get_gateway() -> JavaGateway:
    """
    Gives the gateway with which we can communicate with the SystemDS instance running a
    JMLC (Java Machine Learning Compactor) API.

    :return: the java gateway object
    """
    global JAVA_GATEWAY
    if JAVA_GATEWAY is None:
        if AUTO_START_JMLC_SERVER:
            classpath = os.path.join(_get_module_dir(), 'systemds-java', 'lib', '*')
            process = subprocess.Popen(['java', '-cp', classpath, 'org.tugraz.sysds.pythonapi.PythonDMLScript'],
                                       stdout=subprocess.PIPE)
            process.stdout.readline()  # wait for 'Gateway Server Started\n' written by server
            assert process.poll() is None, "Could not start JMLC server"
        JAVA_GATEWAY = JavaGateway()
    return JAVA_GATEWAY


def create_params_string(unnamed_parameters: Iterable[str], named_parameters: Dict[str, str]) -> str:
    """
    Creates a string for providing parameters in dml. Basically converts both named and unnamed parameter
    to a format which can be used in a dml function call.

    :param unnamed_parameters: the unnamed parameter variables
    :param named_parameters: a dictionary of parameter names and variable names
    :return: the string to represent all parameters
    """
    named_input_strs = (f'{k}={v}' for (k, v) in named_parameters.items())
    return ','.join(chain(unnamed_parameters, named_input_strs))


def _get_module_dir() -> os.PathLike:
    """
    Gives the path to our module

    :return: path to our module
    """
    spec = find_spec(MODULE_NAME)
    return spec.submodule_search_locations[0]
