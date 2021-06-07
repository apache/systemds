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
import json
import os
import socket
import threading
import time
from glob import glob
from queue import Empty, Queue
from subprocess import PIPE, Popen
from threading import Thread
from time import sleep
from typing import Dict, Iterable, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from py4j.java_gateway import GatewayParameters, JavaGateway
from py4j.protocol import Py4JNetworkError
from systemds.operator import (Frame, List, Matrix, OperationNode, Scalar,
                               Source)
from systemds.script_building import OutputType
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.utils.helpers import get_module_dir


class SystemDSContext(object):
    """A context with a connection to a java instance with which SystemDS operations are executed. 
    The java process is started and is running using a random tcp port for instruction parsing.

    This class is used as the starting point for all SystemDS execution. It gives the ability to create
    all the different objects and adding them to the exectution.
    """

    java_gateway: JavaGateway

    def __init__(self, port: int = -1):
        """Starts a new instance of SystemDSContext, in which the connection to a JVM systemds instance is handled
        Any new instance of this SystemDS Context, would start a separate new JVM.

        Standard out and standard error form the JVM is also handled in this class, filling up Queues,
        that can be read from to get the printed statements from the JVM.
        """
        command = self.__build_startup_command()
        process, port = self.__try_startup(command, port)

        # Handle Std out from the subprocess.
        self.__stdout = Queue()
        self.__stderr = Queue()

        self.__stdout_thread = Thread(target=self.__enqueue_output, args=(
            process.stdout, self.__stdout), daemon=True)

        self.__stderr_thread = Thread(target=self.__enqueue_output, args=(
            process.stderr, self.__stderr), daemon=True)

        self.__stdout_thread.start()
        self.__stderr_thread.start()

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

    def exception_and_close(self, e: Exception):
        """
        Method for printing exception, printing stdout and error, while also closing the context correctly.

        :param e: the exception thrown
        """

        # e = sys.exc_info()[0]
        message = "Exception Encountered! closing JVM\n"
        message += "standard out    :\n" + "\n".join(self.get_stdout())
        message += "standard error  :\n" + "\n".join(self.get_stdout())
        message += "Exception       : " + str(e)
        self.close()
        raise RuntimeError(message)

    def __try_startup(self, command, port, rep=0):
        """ Try to perform startup of system.

        :param command: The command to execute for starting JMLC content
        :param port: The port to try to connect to to.
        :param rep: The number of repeated tries to startup the jvm.
        """
        if port == -1:
            assignedPort = self.__get_open_port()
        elif rep == 0:
            assignedPort = port
        else:
            assignedPort = self.__get_open_port()
        fullCommand = []
        fullCommand.extend(command)
        fullCommand.append(str(assignedPort))
        process = Popen(fullCommand, stdout=PIPE, stdin=PIPE, stderr=PIPE)

        try:
            self.__verify_startup(process)

            return process, assignedPort
        except Exception as e:
            self.close()
            if rep > 3:
                raise Exception(
                    "Failed to start SystemDS context with " + str(rep) + " repeated tries")
            else:
                rep += 1
                print("Failed to startup JVM process, retrying: " + str(rep))
                sleep(0.5)
                return self.__try_startup(command, port, rep)

    def __verify_startup(self, process):
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

    def __build_startup_command(self):

        command = ["java", "-cp"]
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

            command.append(classpath)
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
            command.append(lib_cp)

        command.append("org.apache.sysds.api.PythonDMLScript")

        return command

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # no errors to handle to allow continuation
        return None

    def close(self):
        """Close the connection to the java process and do necessary cleanup."""
        if(self.__stdout_thread.is_alive()):
            self.__stdout_thread.join(0)
        if(self.__stdout_thread.is_alive()):
            self.__stderr_thread.join(0)

        pid = self.java_gateway.java_process.pid
        if self.java_gateway.java_gateway_server is not None:
            try:
                self.java_gateway.shutdown(True)
            except Py4JNetworkError as e:
                if "Gateway is not connected" not in str(e):
                    self.java_gateway.java_process.kill()
        os.kill(pid, 14)

    def __enqueue_output(self, out, queue):
        """Method for handling the output from java.
        It is locating the string handeling inside a different thread, since the 'out.readline' is a blocking command.
        """
        for line in iter(out.readline, b""):
            queue.put(line.decode("utf-8").strip())

    def __get_open_port(self):
        """Get a random available port.
        and hope that no other process steals it while we wait for the JVM to startup
        """
        # https://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        return port

    def full(self, shape: Tuple[int, int], value: Union[float, int]) -> 'Matrix':
        """Generates a matrix completely filled with a value

        :param sds_context: SystemDS context
        :param shape: shape (rows and cols) of the matrix TODO tensor
        :param value: the value to fill all cells with
        :return: the OperationNode representing this operation
        """
        unnamed_input_nodes = [value]
        named_input_nodes = {'rows': shape[0], 'cols': shape[1]}
        return Matrix(self, 'matrix', unnamed_input_nodes, named_input_nodes)

    def seq(self, start: Union[float, int], stop: Union[float, int] = None,
            step: Union[float, int] = 1) -> 'Matrix':
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
        return Matrix(self, 'seq', unnamed_input_nodes)

    def rand(self, rows: int, cols: int,
             min: Union[float, int] = None, max: Union[float, int] = None, pdf: str = "uniform",
             sparsity: Union[float, int] = None, seed: Union[float, int] = None,
             lambd: Union[float, int] = 1) -> 'Matrix':
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

        return Matrix(self, 'rand', [], named_input_nodes=named_input_nodes)

    def read(self, path: os.PathLike, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
        """ Read an file from disk. Supportted types include:
        CSV, Matrix Market(coordinate), Text(i,j,v), SystemDS Binary, etc.
        See: http://apache.github.io/systemds/site/dml-language-reference#readwrite-built-in-functions for more details
        :return: an Operation Node, containing the read data the operationNode read can be of types, Matrix, Frame or Scalar.
        """
        mdt_filepath = path + ".mtd"
        if os.path.exists(mdt_filepath):
            with open(mdt_filepath) as jspec_file:
                mtd = json.load(jspec_file)
                kwargs["data_type"] = mtd["data_type"]

        data_type = kwargs.get("data_type", None)
        file_format = kwargs.get("format", None)
        if data_type == "matrix":
            kwargs["data_type"] = f'"{data_type}"'
            return Matrix(self, "read", [f'"{path}"'], named_input_nodes=kwargs)
        elif data_type == "frame":
            kwargs["data_type"] = f'"{data_type}"'
            if isinstance(file_format, str):
                kwargs["format"] = f'"{kwargs["format"]}"'
            return Frame(self, "read", [f'"{path}"'], named_input_nodes=kwargs)
        elif data_type == "scalar":
            kwargs["data_type"] = f'"{data_type}"'
            output_type = OutputType.from_str(kwargs.get("value_type", None))
            kwargs["value_type"] = f'"{output_type.name}"'
            return Scalar(self, "read", [f'"{path}"'], named_input_nodes=kwargs, output_type=output_type)

        print("WARNING: Unknown type read please add a mtd file, or specify in arguments")
        return OperationNode(self, "read", [f'"{path}"'], named_input_nodes=kwargs)

    def scalar(self, v: Dict[str, VALID_INPUT_TYPES]) -> Scalar:
        """ Construct an scalar value, this can contain str, float, double, integers and booleans.
        :return: A scalar containing the given value.
        """
        if type(v) is str:
            if not ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                v = f'"{v}"'

        # output type assign simply assigns the given variable to the value
        # therefore the output type is assign.
        return Scalar(self, v, assign=True, output_type=OutputType.from_str(v))

    def from_numpy(self, mat: np.array,
                   *args: Sequence[VALID_INPUT_TYPES],
                   **kwargs: Dict[str, VALID_INPUT_TYPES]) -> Matrix:
        """Generate DAGNode representing matrix with data given by a numpy array, which will be sent to SystemDS
        on need.

        :param mat: the numpy array
        :param args: unnamed parameters
        :param kwargs: named parameters
        :return: A Matrix
        """

        unnamed_params = ['\'./tmp/{file_name}\'']

        if len(mat.shape) == 2:
            named_params = {'rows': mat.shape[0], 'cols': mat.shape[1]}
        elif len(mat.shape) == 1:
            named_params = {'rows': mat.shape[0], 'cols': 1}
        else:
            # TODO Support tensors.
            raise ValueError("Only two dimensional arrays supported")

        unnamed_params.extend(args)
        named_params.update(kwargs)
        return Matrix(self, 'read', unnamed_params, named_params, local_data=mat)

    def from_pandas(self, df: pd.DataFrame,
                    *args: Sequence[VALID_INPUT_TYPES], **kwargs: Dict[str, VALID_INPUT_TYPES]) -> Frame:
        """Generate DAGNode representing frame with data given by a pandas dataframe, which will be sent to SystemDS
        on need.

        :param df: the pandas dataframe
        :param args: unnamed parameters
        :param kwargs: named parameters
        :return: A Frame
        """
        unnamed_params = ["'./tmp/{file_name}'"]

        if len(df.shape) == 2:
            named_params = {'rows': df.shape[0], 'cols': df.shape[1]}
        elif len(df.shape) == 1:
            named_params = {'rows': df.shape[0], 'cols': 1}
        else:
            # TODO Support tensors.
            raise ValueError("Only two dimensional arrays supported")

        unnamed_params.extend(args)
        named_params["data_type"] = '"frame"'

        self._pd_dataframe = df

        named_params.update(kwargs)
        return Frame(self, "read", unnamed_params, named_params, local_data=df)

    def federated(self, addresses: Iterable[str],
                  ranges: Iterable[Tuple[Iterable[int], Iterable[int]]], *args,
                  **kwargs: Dict[str, VALID_INPUT_TYPES]) -> Matrix:
        """Create federated matrix object.

        :param sds_context: the SystemDS context
        :param addresses: addresses of the federated workers
        :param ranges: for each federated worker a pair of begin and end index of their held matrix
        :param args: unnamed params
        :param kwargs: named params
        :return: The Matrix containing the Federated data.
        """
        addresses_str = 'list(' + \
            ','.join(map(lambda s: f'"{s}"', addresses)) + ')'
        ranges_str = 'list('
        for begin, end in ranges:
            ranges_str += f'list({",".join(map(str, begin))}), list({",".join(map(str, end))}),'
        ranges_str = ranges_str[:-1]
        ranges_str += ')'
        named_params = {'addresses': addresses_str, 'ranges': ranges_str}
        named_params.update(kwargs)
        return Matrix(self, 'federated', args, named_params)

    def source(self, path: str, name: str, print_imported_methods: bool = False) -> Source:
        """Import methods from a given dml file.

        The importing is done thorugh the DML command source, and adds all defined methods from
        the script to the Source object returned in python. This gives the flexibility to call the methods 
        directly on the object returned.

        In systemds a method called func_01 can then be imported using

        ```python
        res = self.sds.source("PATH_TO_FILE", "UNIQUE_NAME").func_01().compute(verbose = True)
        ```

        :param path: The absolute or relative path to the file to import
        :param name: The name to give the imported file in the script, this name must be unique
        :param print_imported_methods: boolean specifying if the imported methods should be printed.
        """
        return Source(self, path, name, print_imported_methods)

    def list(self, *args: Sequence[VALID_INPUT_TYPES], **kwargs: Dict[str, VALID_INPUT_TYPES]) -> List:
        """ Create a List object containing the given nodes.

        Note that only a sequence is allowed, or a dictionary, not both at the same time.
        :param args: A Sequence that will be inserted to a list
        :param kwargs: A Dictionary that will return a dictionary, (internally handled as a list)
        :return: A List 
        """
        return List(self, unnamed_input_nodes=args, named_input_nodes=kwargs)

    def array(self, *args: Sequence[VALID_INPUT_TYPES]) -> List:
        """ Create a List object containing the given nodes.

        Note that only a sequence is allowed, or a dictionary, not both at the same time.
        :param args: A Sequence that will be inserted to a list
        :param kwargs: A Dictionary that will return a dictionary, (internally handled as a list)
        :return: A List 
        """
        return List(self, unnamed_input_nodes=args)

    def dict(self,  **kwargs: Dict[str, VALID_INPUT_TYPES]) -> List:
        """ Create a List object containing the given nodes.

        Note that only a sequence is allowed, or a dictionary, not both at the same time.
        :param args: A Sequence that will be inserted to a list
        :param kwargs: A Dictionary that will return a dictionary, (internally handled as a list)
        :return: A List 
        """
        return List(self, named_input_nodes=kwargs)
