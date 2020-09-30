# -------------------------------------------------------------
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
# -------------------------------------------------------------

__all__ = ["SystemDSContext"]

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
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.utils.helpers import get_module_dir
from systemds.operator import OperationNode
from systemds.script_building import OutputType


class SystemDSContext(object):
    """A context with a connection to a java instance with which SystemDS operations are executed. 
    The java process is started and is running using a random tcp port for instruction parsing."""

    java_gateway: JavaGateway
    __stdout: Queue
    __stderr: Queue

    def __init__(self):
        """Starts a new instance of SystemDSContext, in which the connection to a JVM systemds instance is handled
        Any new instance of this SystemDS Context, would start a separate new JVM.

        Standard out and standard error form the JVM is also handled in this class, filling up Queues,
        that can be read from to get the printed statements from the JVM.
        """

        root = os.environ.get("SYSTEMDS_ROOT")
        if root == None:
            # If there is no systemds install default to use the PIP packaged java files.
            root = os.path.join(get_module_dir(), "systemds-java")

        # nt means its Windows
        cp_separator = ";" if os.name == "nt" else ":"

        if os.environ.get("SYSTEMDS_ROOT") != None:
            lib_cp = os.path.join(root, "target", "lib", "*")
            systemds_cp = os.path.join(root, "target", "SystemDS.jar")
            classpath = cp_separator.join([lib_cp, systemds_cp])

            command = ["java", "-cp", classpath]
            files = glob(os.path.join(root, "conf", "log4j*.properties"))
            if len(files) > 1:
                print(
                    "WARNING: Multiple logging files found selecting: " + files[0])
            if len(files) == 0:
                print("WARNING: No log4j file found at: "
                      + os.path.join(root, "conf")
                      + " therefore using default settings")
            else:
                command.append("-Dlog4j.configuration=file:" + files[0])
        else:
            lib_cp = os.path.join(root, "lib", "*")
            command = ["java", "-cp", lib_cp]

        command.append("org.apache.sysds.api.PythonDMLScript")

        # TODO add an argument parser here

        # Find a random port, and hope that no other process
        # steals it while we wait for the JVM to startup
        port = self.__get_open_port()
        command.append(str(port))

        process = Popen(command, stdout=PIPE, stdin=PIPE, stderr=PIPE)
        first_stdout = process.stdout.readline()

        if(not b"GatewayServer Started" in first_stdout):
            stderr = process.stderr.readline().decode("utf-8")
            if(len(stderr) > 1):
                raise Exception(
                    "Exception in startup of GatewayServer: " + stderr)
            outputs = []
            outputs.append(first_stdout.decode("utf-8"))
            max_tries = 10
            for i in range(max_tries):
                next_line = process.stdout.readline()
                if(b"GatewayServer Started" in next_line):
                    print("WARNING: Stdout corrupted by prints: " + str(outputs))
                    print("Startup success")
                    break
                else:
                    outputs.append(next_line)

                if (i == max_tries-1):
                    raise Exception("Error in startup of systemDS gateway process: \n gateway StdOut: " + str(
                        outputs) + " \n gateway StdErr" + process.stderr.readline().decode("utf-8"))

        # Handle Std out from the subprocess.
        self.__stdout = Queue()
        self.__stderr = Queue()

        Thread(target=self.__enqueue_output, args=(
            process.stdout, self.__stdout), daemon=True).start()
        Thread(target=self.__enqueue_output, args=(
            process.stderr, self.__stderr), daemon=True).start()

        # Py4j connect to the started process.
        gwp = GatewayParameters(port=port, eager_load=True)
        self.java_gateway = JavaGateway(
            gateway_parameters=gwp, java_process=process)

    def get_stdout(self, lines: int = -1):
        """Getter for the stdout of the java subprocess
        The output is taken from the stdout queue and returned in a new list.
        :param lines: The number of lines to try to read from the stdout queue.
        default -1 prints all current lines in the queue.
        """
        if lines == -1 or self.__stdout.qsize() < lines:
            return [self.__stdout.get() for x in range(self.__stdout.qsize())]
        else:
            return [self.__stdout.get() for x in range(lines)]

    def get_stderr(self, lines: int = -1):
        """Getter for the stderr of the java subprocess
        The output is taken from the stderr queue and returned in a new list.
        :param lines: The number of lines to try to read from the stderr queue.
        default -1 prints all current lines in the queue.
        """
        if lines == -1 or self.__stderr.qsize() < lines:
            return [self.__stderr.get() for x in range(self.__stderr.qsize())]
        else:
            return [self.__stderr.get() for x in range(lines)]

    def exception_and_close(self, e):
        """
        Method for printing exception, printing stdout and error, while also closing the context correctly.
        """
        # e = sys.exc_info()[0]
        print("Exception Encountered! closing JVM")
        print("standard out:")
        [print(x) for x in self.get_stdout()]
        print("standard error")
        [print(x) for x in self.get_stderr()]
        print("exception")
        print(e)
        self.close()
        exit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # no errors to handle to allow continuation
        return None

    def close(self):
        """Close the connection to the java process and do necessary cleanup."""
        process: Popen = self.java_gateway.java_process
        self.java_gateway.shutdown()
        # Send SigTerm
        os.kill(process.pid, 14)

    def __enqueue_output(self, out, queue):
        """Method for handling the output from java.
        It is locating the string handeling inside a different thread, since the 'out.readline' is a blocking command.
        """
        for line in iter(out.readline, b""):
            queue.put(line.decode("utf-8").strip())

    def __get_open_port(self):
        """Get a random available port."""
        # TODO Verify that it is not taking some critical ports change to select a good port range.
        # TODO If it tries to select a port already in use, find another.
        # https://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        return port

    def full(self, shape: Tuple[int, int], value: Union[float, int]) -> 'OperationNode':
        """Generates a matrix completely filled with a value

        :param sds_context: SystemDS context
        :param shape: shape (rows and cols) of the matrix TODO tensor
        :param value: the value to fill all cells with
        :return: the OperationNode representing this operation
        """
        unnamed_input_nodes = [value]
        named_input_nodes = {'rows': shape[0], 'cols': shape[1]}
        return OperationNode(self, 'matrix', unnamed_input_nodes, named_input_nodes)

    def seq(self, start: Union[float, int], stop: Union[float, int] = None,
            step: Union[float, int] = 1) -> 'OperationNode':
        """Create a single column vector with values from `start` to `stop` and an increment of `step`.
        If no stop is defined and only one parameter is given, then start will be 0 and the parameter will be interpreted as
        stop.

        :param sds_context: SystemDS context
        :param start: the starting value
        :param stop: the maximum value
        :param step: the step size
        :return: the OperationNode representing this operation
        """
        if stop is None:
            stop = start
            start = 0
        unnamed_input_nodes = [start, stop, step]
        return OperationNode(self, 'seq', unnamed_input_nodes)

    def rand(self, rows: int, cols: int,
             min: Union[float, int] = None, max: Union[float, int] = None, pdf: str = "uniform",
             sparsity: Union[float, int] = None, seed: Union[float, int] = None,
             lambd: Union[float, int] = 1) -> 'OperationNode':
        """Generates a matrix filled with random values

        :param sds_context: SystemDS context
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
        available_pdfs = ["uniform", "normal", "poisson"]
        if rows < 0:
            raise ValueError("In rand statement, can only assign rows a long (integer) value >= 0 "
                             "-- attempted to assign value: {r}".format(r=rows))
        if cols < 0:
            raise ValueError("In rand statement, can only assign cols a long (integer) value >= 0 "
                             "-- attempted to assign value: {c}".format(c=cols))
        if pdf not in available_pdfs:
            raise ValueError("The pdf passed is invalid! given: {g}, expected: {e}".format(
                g=pdf, e=available_pdfs))

        pdf = '\"' + pdf + '\"'
        named_input_nodes = {
            'rows': rows, 'cols': cols, 'pdf': pdf, 'lambda': lambd}
        if min is not None:
            named_input_nodes['min'] = min
        if max is not None:
            named_input_nodes['max'] = max
        if sparsity is not None:
            named_input_nodes['sparsity'] = sparsity
        if seed is not None:
            named_input_nodes['seed'] = seed

        return OperationNode(self, 'rand', [], named_input_nodes=named_input_nodes)

    def read(self, path: os.PathLike, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> 'OperationNode':
        """ Read an file from disk. Supportted types include:
        CSV, Matrix Market(coordinate), Text(i,j,v), SystemDS Binay
        See: http://apache.github.io/systemds/site/dml-language-reference#readwrite-built-in-functions for more details
        :return: an Operation Node, containing the read data.
        """
        return OperationNode(self, 'read', [f'"{path}"'], named_input_nodes=kwargs, shape=(-1,))

    def scalar(self, v: Dict[str, VALID_INPUT_TYPES]) -> 'OperationNode':
        """ Construct an scalar value, this can contain str, float, double, integers and booleans.
        :return: An `OperationNode` containing the scalar value.
        """
        if type(v) is str:
            if not ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                v = f'"{v}"'
        # output type assign simply assigns the given variable to the value
        # therefore the output type is assign.
        return OperationNode(self, v, output_type=OutputType.ASSIGN)
