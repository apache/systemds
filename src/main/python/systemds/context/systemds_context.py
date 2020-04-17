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

__all__ = ["SystemDSContext"]

import os
import subprocess
import threading
from typing import Optional, Sequence, Union, Dict, Tuple, Iterable

import numpy as np
from py4j.java_gateway import JavaGateway
from py4j.protocol import Py4JNetworkError

from systemds.matrix import full, seq, federated, Matrix, rand, OperationNode
from systemds.utils.helpers import get_module_dir
from systemds.utils.consts import VALID_INPUT_TYPES

PROCESS_LOCK: threading.Lock = threading.Lock()
PROCESS: Optional[subprocess.Popen] = None
ACTIVE_PROCESS_CONNECTIONS: int = 0


class SystemDSContext(object):
    """A context with a connection to the java instance with which SystemDS operations are executed.
    If necessary this class might also start a java process which is used for the SystemDS operations,
    before connecting."""
    _java_gateway: Optional[JavaGateway]

    def __init__(self):
        global PROCESS_LOCK
        global PROCESS
        global ACTIVE_PROCESS_CONNECTIONS
        # make sure that a process is only started if necessary and no other thread
        # is killing the process while the connection is established
        PROCESS_LOCK.acquire()
        try:
            # attempt connection to manually started java instance
            self._java_gateway = JavaGateway(eager_load=True)
        except Py4JNetworkError:
            # if no java instance is running start it
            systemds_java_path = os.path.join(get_module_dir(), 'systemds-java')
            cp_separator = ':'
            if os.name == 'nt':  # nt means its Windows
                cp_separator = ';'
            lib_cp = os.path.join(systemds_java_path, 'lib', '*')
            systemds_cp = os.path.join(systemds_java_path, '*')
            classpath = cp_separator.join([lib_cp, systemds_cp])
            process = subprocess.Popen(['java', '-cp', classpath, 'org.apache.sysds.pythonapi.PythonDMLScript'],
                                       stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            process.stdout.readline()  # wait for 'Gateway Server Started\n' written by server
            assert process.poll() is None, "Could not start JMLC server"
            self._java_gateway = JavaGateway()
            PROCESS = process
        if PROCESS is not None:
            ACTIVE_PROCESS_CONNECTIONS += 1
        PROCESS_LOCK.release()

    @property
    def java_gateway(self):
        return self._java_gateway

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # no errors to handle to allow continuation
        return None

    def close(self):
        """Close the connection to the java process and do necessary cleanup."""

        global PROCESS_LOCK
        global PROCESS
        global ACTIVE_PROCESS_CONNECTIONS
        self._java_gateway.shutdown()
        PROCESS_LOCK.acquire()
        # check if no other thread is connected to the process, if it was started as a subprocess
        if PROCESS is not None and ACTIVE_PROCESS_CONNECTIONS == 1:
            # stop the process by sending a new line (it will shutdown on its own)
            PROCESS.communicate(input=b'\n')
            PROCESS.wait()
            PROCESS = None
            ACTIVE_PROCESS_CONNECTIONS = 0
        PROCESS_LOCK.release()

    def matrix(self, mat: Union[np.array, os.PathLike], *args: Sequence[VALID_INPUT_TYPES],
               **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'Matrix':
        """ Create matrix.

        :param mat: Matrix given by numpy array or path to matrix file
        :param args: additional arguments
        :param kwargs: additional named arguments
        :return: the OperationNode representing this operation
        """
        return Matrix(self, mat, *args, **kwargs)

    def federated(self, addresses: Iterable[str], ranges: Iterable[Tuple[Iterable[int], Iterable[int]]],
                  *args: Sequence[VALID_INPUT_TYPES], **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'OperationNode':
        """Create federated matrix object.
    
        :param addresses: addresses of the federated workers
        :param ranges: for each federated worker a pair of begin and end index of their held matrix
        :param args: unnamed params
        :param kwargs: named params
        :return: the OperationNode representing this operation
        """
        return federated(self, addresses, ranges, *args, **kwargs)

    def full(self, shape: Tuple[int, int], value: Union[float, int]) -> 'OperationNode':
        """Generates a matrix completely filled with a value

        :param shape: shape (rows and cols) of the matrix
        :param value: the value to fill all cells with
        :return: the OperationNode representing this operation
        """
        return full(self, shape, value)

    def seq(self, start: Union[float, int], stop: Union[float, int] = None,
            step: Union[float, int] = 1) -> 'OperationNode':
        """Create a single column vector with values from `start` to `stop` and an increment of `step`.
        If no stop is defined and only one parameter is given, then start will be 0 and the parameter will be
        interpreted as stop.

        :param start: the starting value
        :param stop: the maximum value
        :param step: the step size
        :return: the OperationNode representing this operation
        """
        return seq(self, start, stop, step)

    def rand(self, rows: int, cols: int, min: Union[float, int] = None,
             max: Union[float, int] = None, pdf: str = "uniform",
             sparsity: Union[float, int] = None, seed: Union[float, int] = None,
             lambd: Union[float, int] = 1) -> OperationNode:
        """Generates a matrix filled with random values

        :param rows: number of rows
        :param cols: number of cols
        :param min: min value for cells
        :param max: max value for cells
        :param pdf: "uniform"/"normal"/"poison" distribution
        :param sparsity: fraction of non-zero cells
        :param seed: random seed
        :param lambd: lamda value for "poison" distribution
        :return:
        """
        return rand(self, rows, cols, min, max, pdf, sparsity, seed, lambd)