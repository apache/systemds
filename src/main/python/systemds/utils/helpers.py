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
import subprocess
from itertools import chain
from typing import Iterable, Dict
from importlib.util import find_spec

from py4j.java_gateway import JavaGateway
from py4j.protocol import Py4JNetworkError

JAVA_GATEWAY = None
MODULE_NAME = 'systemds'


def get_gateway() -> JavaGateway:
    """
    Gives the gateway with which we can communicate with the SystemDS instance running a
    JMLC (Java Machine Learning Compactor) API.

    :return: the java gateway object
    """
    global JAVA_GATEWAY
    if JAVA_GATEWAY is None:
        try:
            JAVA_GATEWAY = JavaGateway(eager_load=True)
        except Py4JNetworkError:  # if no java instance is running start it
            systemds_java_path = os.path.join(_get_module_dir(), 'systemds-java')
            cp_separator = ':'
            if os.name == 'nt':  # nt means its Windows
                cp_separator = ';'
            lib_cp = os.path.join(systemds_java_path, 'lib', '*')
            systemds_cp = os.path.join(systemds_java_path, '*')
            classpath = cp_separator.join([lib_cp, systemds_cp])
            process = subprocess.Popen(['java', '-cp', classpath, 'org.tugraz.sysds.pythonapi.PythonDMLScript'],
                                       stdout=subprocess.PIPE)
            print(process.stdout.readline())  # wait for 'Gateway Server Started\n' written by server
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
