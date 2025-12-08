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

import json
import logging
import os
import uuid
import socket
import sys
import struct
import traceback
from contextlib import contextmanager
from glob import glob
from queue import Queue
from subprocess import PIPE, Popen
from threading import Thread
from time import sleep, time
from typing import Dict, Iterable, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from py4j.java_gateway import GatewayParameters, JavaGateway, Py4JNetworkError
from systemds.operator import (
    Frame,
    List,
    Matrix,
    OperationNode,
    Scalar,
    Source,
    Combine,
    MultiReturn,
)
from systemds.script_building import DMLScript
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.utils.helpers import get_module_dir, valuetype_from_str


class SystemDSContext(object):
    """A context with a connection to a java instance with which SystemDS operations are executed.
    The java process is started and is running using a random tcp port for instruction parsing.

    This class is used as the starting point for all SystemDS execution. It gives the ability to create
    all the different objects and adding them to the execution.
    """

    java_gateway: JavaGateway
    _capture_statistics: bool = False
    _statistics: str = ""
    _log: logging.Logger
    __stdout: Queue = None
    __stderr: Queue = None
    _FIFO_PATH = "/tmp/systemds/pipes/"
    _FIFO_PY2JAVA_BASE = "py2java"
    _FIFO_JAVA2PY_BASE = "java2py"
    _FIFO_PY2JAVA_PIPES = []
    _FIFO_JAVA2PY_PIPES = []
    _data_transfer_mode = 0
    _multi_pipe_enabled = False
    _logging_initialized = False
    _executor_pool = ThreadPoolExecutor(max_workers=os.cpu_count() * 2 or 4)

    def __init__(
        self,
        port: int = -1,
        capture_statistics: bool = False,
        capture_stdout: bool = False,
        logging_level: int = 30,
        py4j_logging_level: int = 50,
        data_transfer_mode: int = 1,
        multi_pipe_enabled: bool = False,
    ):
        """Starts a new instance of SystemDSContext, in which the connection to a JVM systemds instance is handled
        Any new instance of this SystemDS Context, would start a separate new JVM.

        Standard out and standard error form the JVM is also handled in this class, filling up Queues,
        that can be read from to get the printed statements from the JVM.

        :param port: default -1, giving a random port for communication with JVM
        :param capture_statistics: If the statistics of the execution in SystemDS should be captured
        :param capture_stdout: If the standard out should be captured in Java SystemDS and maintained in ques
        :param logging_level: Specify the logging level used for informative messages, default 20 indicating INFO.
            The logging levels are as follows: 10 DEBUG, 20 INFO, 30 WARNING, 40 ERROR, 50 CRITICAL.
        :param py4j_logging_level: The logging level for Py4j to use, since all communication to the JVM is done through this,
            it can be verbose if not set high.
        :param data_transfer_mode: default 0,
        """

        self.__setup_logging(logging_level, py4j_logging_level)
        self.__start(port, capture_stdout)
        self.capture_stats(capture_statistics)
        self._log.debug("Started JVM and SystemDS python context manager")
        self.__setup_data_transfer(data_transfer_mode, multi_pipe_enabled)

    def __setup_data_transfer(self, data_transfer_mode=0, multi_pipe_enabled=False):
        self._data_transfer_mode = data_transfer_mode
        self._multi_pipe_enabled = multi_pipe_enabled
        if os.name == "posix" and data_transfer_mode == 1:
            num_pipes = (os.cpu_count() or 2) if multi_pipe_enabled else 1
            self.__make_fifo_named_pipes(num_pipes)
            in_pipes, out_pipes = self.__init_pipes(num_pipes)

            self._log.info(
                "Data transfer: Handshake done for {} IN / OUT Pipes".format(num_pipes)
            )
            self._FIFO_PY2JAVA_PIPES = out_pipes
            self._FIFO_JAVA2PY_PIPES = in_pipes
        else:
            self._log.info("Using py4j for data transfer")
            self._data_transfer_mode = 0

    def __init_pipes(self, num_pipes):

        def open_pipe_write(path, pipe_id):
            self._log.debug("Opening {} for writing in Py".format(path))
            pipe = open(path, "wb")
            handshake = struct.pack("<i", pipe_id + 1000)
            os.write(pipe.fileno(), handshake)
            return pipe

        def open_pipe_read(path, expected_id):
            expected_id += 1000
            self._log.debug("Opening {} for reading in Py".format(path))
            pipe = open(path, "rb")
            handshake = os.read(pipe.fileno(), 4)
            received_id = struct.unpack("<i", handshake)[0]
            if received_id != expected_id:
                self._log.debug(
                    "Mismatch: {}, expected: {}".format(received_id, expected_id)
                )
            else:
                self._log.debug("JAVA2PY pipe {} is ready!".format(expected_id - 1000))
            return pipe

        def open_pipes_java(ep, p, n):
            ep.openPipes(p, n)

        fut = self._executor_pool.submit(
            open_pipes_java, self.java_gateway.entry_point, self._FIFO_PATH, num_pipes
        )
        out_f, in_f = ([], [])
        for i in range(num_pipes):
            out_f.append(
                self._executor_pool.submit(
                    open_pipe_write,
                    "{}/{}-{}".format(self._FIFO_PATH, self._FIFO_PY2JAVA_BASE, i),
                    i,
                )
            )
            in_f.append(
                self._executor_pool.submit(
                    open_pipe_read,
                    "{}/{}-{}".format(self._FIFO_PATH, self._FIFO_JAVA2PY_BASE, i),
                    i,
                )
            )
        fut.result()
        out_pipes = [f.result() for f in out_f]
        in_pipes = [f.result() for f in in_f]
        return in_pipes, out_pipes

    def __delete_tmp_files(self):
        if os.path.isdir(self._FIFO_PATH):
            for root, dirs, files in os.walk(self._FIFO_PATH):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
            os.rmdir(self._FIFO_PATH)
            self._log.debug("Cleaned up tmp files")
        else:
            self._log.debug("FIFO Path does not exist")

    def __make_fifo_named_pipes(self, num_pipes):
        self._FIFO_PATH += str(uuid.uuid4())
        os.makedirs(self._FIFO_PATH, exist_ok=True)
        direction, name, i = (None, None, 0)
        directions = [self._FIFO_PY2JAVA_BASE, self._FIFO_JAVA2PY_BASE]
        try:
            for direction in directions:
                for i in range(num_pipes):
                    name = direction + "-{}".format(i)
                    os.mkfifo(self._FIFO_PATH + "/" + name)
        except IOError:
            # clean up already created pipes in self.FIFO_PATH
            self.__delete_tmp_files()
            raise Exception("Creating named pipe {} failed".format(name))

    def get_stdout(self, lines: int = -1):
        """Getter for the stdout of the java subprocess
        The output is taken from the stdout queue and returned in a new list.
        :param lines: The number of lines to try to read from the stdout queue.
        default -1 prints all current lines in the queue.
        """
        if self.__stdout:
            if lines == -1 or self.__stdout.qsize() < lines:
                return [self.__stdout.get() for x in range(self.__stdout.qsize())]
            else:
                return [self.__stdout.get() for x in range(lines)]
        else:
            return []

    def get_stderr(self, lines: int = -1):
        """Getter for the stderr of the java subprocess
        The output is taken from the stderr queue and returned in a new list.
        :param lines: The number of lines to try to read from the stderr queue.
        default -1 prints all current lines in the queue.
        """
        if self.__stderr:
            if lines == -1 or self.__stderr.qsize() < lines:
                return [self.__stderr.get() for x in range(self.__stderr.qsize())]
            else:
                return [self.__stderr.get() for x in range(lines)]
        else:
            return []

    def exception_and_close(self, exception, trace_back_limit: int = None):
        """
        Method for printing exception, printing stdout and error, while also closing the context correctly.

        :param e: the exception thrown
        """

        message = ""
        stdOut = self.get_stdout()
        if stdOut:
            message += "standard out    :\n" + "\n".join(stdOut)
        stdErr = self.get_stderr()
        if stdErr:
            message += "standard error  :\n" + "\n".join(stdErr)
        message += "\n\n"
        message += str(exception)
        sys.tracebacklimit = trace_back_limit

        # print error in case we encounter another exception during closing
        # also print stacktrace of the original exception
        tb_str = (
            exception
            if isinstance(exception, str)
            else "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
        )
        self._log.error(
            "Encountered exception & shutting down: \n\n{}".format(tb_str.strip())
        )

        self.close()
        raise RuntimeError(message)

    def __try_startup(self, command: str, capture_stdout: bool) -> Popen:
        if capture_stdout:
            process = Popen(command, stdout=PIPE, stdin=PIPE, stderr=PIPE)

            # Handle Std out from the subprocess.
            self.__stdout = Queue()
            self.__stderr = Queue()

            self.__stdout_thread = Thread(
                target=self.__enqueue_output,
                args=(process.stdout, self.__stdout),
                daemon=True,
            )

            self.__stderr_thread = Thread(
                target=self.__enqueue_output,
                args=(process.stderr, self.__stderr),
                daemon=True,
            )

            self.__stdout_thread.start()
            self.__stderr_thread.start()
            return process
        else:
            return Popen(command)

    def __build_startup_command(self, port: int):
        """Build the command line argument for the startup of the JVM
        :param port: The port address to use if -1 chose random port."""

        # Base command
        command = ["java", "--add-modules=jdk.incubator.vector", "-cp"]

        # Find the operating system specifc separator, nt means its Windows
        cp_separator = ";" if os.name == "nt" else ":"
        root = os.environ.get("SYSTEMDS_ROOT")

        if root == None:
            # If there is no systemds install default to use the pip packaged java files.
            root = os.path.join(get_module_dir())

        # Find the SystemDS jar file.
        if root != None:  # root path was set
            self._log.debug("SYSTEMDS_ROOT was set, searching for jar file")
            lib_release = os.path.join(root, "lib")
            systemds_cp = os.path.join(root, "target", "SystemDS.jar")
            if os.path.exists(
                lib_release
            ):  # It looks like it was a release path for root.
                classpath = os.path.join(root, "SystemDS.jar")
                if not os.path.exists(classpath):
                    for f in os.listdir(root):
                        if "systemds" in f:
                            if os.path.exists(classpath):
                                raise (
                                    ValueError(
                                        "Invalid setup there were multiple conflicting systemds jar fines in"
                                        + root
                                    )
                                )
                            else:
                                classpath = os.path.join(root, f)
                    if not os.path.exists(classpath):
                        raise ValueError(
                            "Invalid setup did not find SystemDS jar file in " + root
                        )
            elif os.path.exists(systemds_cp):
                classpath = cp_separator.join([systemds_cp])
            else:
                raise ValueError(
                    "Invalid setup at SYSTEMDS_ROOT env variable path " + root
                )
        else:  # root path was not set use the pip installed SystemDS
            self._log.warning(
                "SYSTEMDS_ROOT was unset, defaulting to python packaged jar files"
            )
            systemds_cp = os.path.join(root, "SystemDS.jar")
            classpath = cp_separator.join([systemds_cp])

        command.append(classpath)

        # Find the logging configuration file.
        if os.environ.get("LOG4JPROP") == None:
            files = glob(os.path.join(root, "conf", "log4j*.properties"))
            if len(files) > 1:
                self._log.warning("Multiple logging files found selecting: " + files[0])
            if len(files) == 0:
                self._log.warning(
                    "No log4j file found at: "
                    + os.path.join(root, "conf")
                    + " therefore using default settings"
                )
            else:
                command.append("-Dlog4j.configuration=file:" + files[0])
        else:
            logging_file = os.environ.get("LOG4JPROP")
            if os.path.exists(logging_file):
                command.append(
                    "-Dlog4j.configuration=file:" + os.environ.get("LOG4JPROP")
                )
            else:
                self._log.warning(
                    "LOG4JPROP is set but path is invalid: " + str(logging_file)
                )

        # Specify the main function inside SystemDS to launch in java.
        command.append("org.apache.sysds.api.PythonDMLScript")

        # Find the configuration file for systemds.
        # TODO: refine the choice of configuration file
        files = glob(os.path.join(root, "conf", "SystemDS*.xml"))
        if len(files) > 1:
            self._log.warning("Multiple config files found selecting: " + files[0])
        if len(files) == 0:
            self._log.warning(
                "No xml config file found at: "
                + os.path.join(root, "conf")
                + " therefore using default settings"
            )
        else:
            command.append("-config")
            command.append(files[0])

        if port == -1:
            actual_port = self.__get_open_port()
        else:
            actual_port = port

        command.append("--python")
        command.append(str(actual_port))

        self._log.info("Command " + str(command))
        self._log.info("Port used for communication: " + str(actual_port))

        return command, actual_port

    def __start(self, port: int, capture_stdout: bool, retry: int = 0):
        """Starts the JVM and establishes connection to it via calls to:
        jvm.System.currentTimeMillis()
        If the connection is not established the connection is
        tried to be established multiple times,
        and if this fails new JVMs are allocated up to a total of 3 times.

        :param port: The port to try, if -1 chose random.
        :param capture_stdout: If the output of the JVM should be captured.
        :param retry: The Retry number of the current startup.
        """
        if retry > 3:
            raise Exception("Failed startup of SystemDS Context with 3 repeats")

        if port != -1 and self.__is_port_in_use(port):
            port = -1
        command, actual_port = self.__build_startup_command(port)

        # Verify the port intended is available.
        while self.__is_port_in_use(actual_port):
            (
                command,
                actual_port,
            ) = self.__build_startup_command(actual_port)

        process = self.__try_startup(command, capture_stdout)

        # Eager load to test connection to the JVM via calls to currentTimeMillis
        gwp = GatewayParameters(port=actual_port, eager_load=True)
        try:
            connect_retry = 1
            max_retry = 25
            while connect_retry < max_retry:
                # Increasing sleep time to establish connection.
                sleep_time = 0.04 * connect_retry
                sleep(sleep_time)
                try:
                    self.java_gateway = JavaGateway(
                        gateway_parameters=gwp, java_process=process
                    )
                    # Successful startup.
                    return
                except Py4JNetworkError as pe:
                    m = str(pe)
                    if (
                        "An error occurred while trying to connect to the Java server"
                        in m
                    ):
                        # Here the startup failed because the java process is not ready.
                        connect_retry += 1
                    else:  # unknown new error or java process crashed
                        raise pe
                except Exception as e:
                    raise Exception(
                        "Exception hit when connecting to JavaGateway, perhaps the JVM terminated because of port",
                        e,
                    )
            raise Exception("Failed to connect to process, making a new JVM")
        except Exception:
            self.__kill_Popen(process)
            self.__start(-1, capture_stdout, retry + 1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # no errors to handle to allow continuation
        return None

    def close(self):
        """Close the connection to the java process and do necessary cleanup."""
        if hasattr(self, "java_gateway"):
            if self._data_transfer_mode == 1:
                self._log.debug("Closing all Pipes in Python")
                for pipe in self._FIFO_JAVA2PY_PIPES:
                    pipe.close()
                self._FIFO_JAVA2PY_PIPES = []
                for pipe in self._FIFO_PY2JAVA_PIPES:
                    pipe.close()
                self._FIFO_PY2JAVA_PIPES = []
                self._log.debug("All Pipes are closed in Python")

                if self.java_gateway._gateway_client.is_connected:
                    self._log.debug("Closing all Pipes in Java")
                    self.java_gateway.entry_point.closePipes()
                    self._log.debug("All Pipes are closed in Java")
                else:
                    self._log.debug("Java Gateway is not connected anymore")
                self.__delete_tmp_files()

            self.java_gateway.shutdown()
            self.__kill_Popen(self.java_gateway.java_process)
        if hasattr(self, "__process"):
            logging.error("Has process variable")
            self.__kill_Popen(self.__process)
        if hasattr(self, "__stdout_thread") and self.__stdout_thread.is_alive():
            self.__stdout_thread.join(0)
            for line in self.get_stdout():
                print(line)
        if hasattr(self, "__stderr_thread") and self.__stderr_thread.is_alive():
            self.__stderr_thread.join(0)
            for line in self.get_stderr():
                print(line)

    def __kill_Popen(self, process: Popen):
        """Stop the process at the Popen.
        :param process: The process to stop"""
        process.kill()
        process.__exit__(None, None, None)

    def __enqueue_output(self, out, queue):
        """Method for handling the output from java.
        It is locating the string handling inside a different thread, since the 'out.readline' is a blocking command.
        """
        for line in iter(out.readline, b""):
            line_string = line.decode("utf-8")
            queue.put(line_string.strip())

    def __get_open_port(self):
        """Get a random available port.
        and hope that no other process steals it while we wait for the JVM to startup
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            return s.getsockname()[1]

    def __is_port_in_use(self, port: int) -> bool:
        """Get if the given port is in use.
        :param port: The port to analyze"""
        # https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _execution_completed(self, script: DMLScript):
        """
        Should/will be called after execution of a script.
        Used to update statistics.
        :param script: The script that got executed
        """
        if self._capture_statistics:
            self._statistics += script.prepared_script.statistics()

    def capture_stats(self, enable: bool = True):
        """
        Enable (or disable) capturing of execution statistics.
        :param enable: if `True` enable capturing, else disable it
        """
        self._capture_statistics = enable
        self.java_gateway.entry_point.getConnection().setStatistics(enable)

    @contextmanager
    def capture_stats_context(self):
        """
        Context for capturing statistics. Should be used in a `with` statement.
        Afterwards capturing will be reset to the state it was before.

        Example:

        # ```Python
        # with sds.capture_stats_context():
        #     a = some_computation.compute()
        #     b = another_computation.compute()
        # print(sds.take_stats())
        # ```

        :return: a context object to be used in a `with` statement
        """
        was_enabled = self._capture_statistics
        try:
            self.capture_stats(True)
            yield None
        finally:
            self.capture_stats(was_enabled)

    def get_stats(self):
        """Get the captured statistics. Will not clear the captured statistics.

        See `take_stats()` for an option that also clears the captured statistics.
        :return: The captured statistics
        """
        return self._statistics

    def take_stats(self):
        """Get the captured statistics and clear the captured statistics.

        See `get_stats()` for an option that does not clear the captured statistics.
        :return: The captured statistics
        """
        stats = self.get_stats()
        self.clear_stats()
        return stats

    def clear_stats(self):
        """Clears the captured statistics."""
        self._statistics = ""

    def full(self, shape: Tuple[int, int], value: Union[float, int]) -> "Matrix":
        """Generates a matrix completely filled with a value

        :param sds_context: SystemDS context
        :param shape: shape (rows and cols) of the matrix TODO tensor
        :param value: the value to fill all cells with
        :return: the OperationNode representing this operation
        """
        unnamed_input_nodes = [value]
        named_input_nodes = {"rows": shape[0], "cols": shape[1]}
        return Matrix(self, "matrix", unnamed_input_nodes, named_input_nodes)

    def seq(
        self,
        start: Union[float, int],
        stop: Union[float, int] = None,
        step: Union[float, int] = 1,
    ) -> "Matrix":
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
        return Matrix(self, "seq", unnamed_input_nodes)

    def rand(
        self,
        rows: int,
        cols: int,
        min: Union[float, int] = None,
        max: Union[float, int] = None,
        pdf: str = "uniform",
        sparsity: Union[float, int] = None,
        seed: Union[float, int] = None,
        lamb: Union[float, int] = 1,
    ) -> "Matrix":
        """Generates a matrix filled with random values

        :param sds_context: SystemDS context
        :param rows: number of rows
        :param cols: number of cols
        :param min: min value for cells
        :param max: max value for cells
        :param pdf: probability distribution function: "uniform"/"normal"/"poison" distribution
        :param sparsity: fraction of non-zero cells
        :param seed: random seed
        :param lamb: lambda value for "poison" distribution
        :return:
        """
        available_pdf = ["uniform", "normal", "poisson"]
        if rows < 0:
            raise ValueError(
                "In rand statement, can only assign rows a long (integer) value >= 0 "
                "-- attempted to assign value: {r}".format(r=rows)
            )
        if cols < 0:
            raise ValueError(
                "In rand statement, can only assign cols a long (integer) value >= 0 "
                "-- attempted to assign value: {c}".format(c=cols)
            )
        if pdf not in available_pdf:
            raise ValueError(
                "The pdf passed is invalid! given: {g}, expected: {e}".format(
                    g=pdf, e=available_pdf
                )
            )

        pdf = '"' + pdf + '"'
        named_input_nodes = {"rows": rows, "cols": cols, "pdf": pdf, "lambda": lamb}
        if min is not None:
            named_input_nodes["min"] = min
        if max is not None:
            named_input_nodes["max"] = max
        if sparsity is not None:
            named_input_nodes["sparsity"] = sparsity
        if seed is not None:
            named_input_nodes["seed"] = seed

        return Matrix(self, "rand", [], named_input_nodes=named_input_nodes)

    def __fix_string_args(self, arg: str) -> str:
        nf = str(arg).replace('"', "").replace("'", "")
        return f'"{nf}"'

    def read(
        self, path: os.PathLike, **kwargs: Dict[str, VALID_INPUT_TYPES]
    ) -> OperationNode:
        """Read an file from disk. Supported types include:
        CSV, Matrix Market(coordinate), Text(i,j,v), SystemDS Binary, etc.
        See: http://apache.github.io/systemds/site/dml-language-reference#readwrite-built-in-functions for more details
        :return: an Operation Node, containing the read data the operationNode read can be of types, Matrix, Frame or Scalar.
        """
        mdt_filepath = path + ".mtd"
        # If metadata file is existing, then simply use that and force data type to mtd file
        if os.path.exists(mdt_filepath):
            with open(mdt_filepath) as jspec_file:
                mtd = json.load(jspec_file)
                kwargs["data_type"] = mtd["data_type"]
            if kwargs.get("format", None):
                kwargs["format"] = self.__fix_string_args(kwargs["format"])
        elif kwargs.get("format", None):  # If format is specified. Then use that format
            kwargs["format"] = self.__fix_string_args(kwargs["format"])
        else:  # Otherwise guess at what format the file is based on file extension
            if ".csv" in path[-4:]:
                kwargs["format"] = '"csv"'
                self._log.warning(
                    "Guessing '"
                    + path
                    + "' is a csv file, please add a mtd file, or specify in arguments"
                )
                if (
                    not ("header" in kwargs)
                    and "data_type" in kwargs
                    and kwargs["data_type"] == "frame"
                ):
                    kwargs["header"] = True

        data_type = kwargs.get("data_type", None)

        if data_type == "matrix":
            kwargs["data_type"] = f'"{data_type}"'
            return Matrix(self, "read", [f'"{path}"'], named_input_nodes=kwargs)
        elif data_type == "frame":
            kwargs["data_type"] = f'"{data_type}"'
            return Frame(self, "read", [f'"{path}"'], named_input_nodes=kwargs)
        elif data_type == "scalar":
            kwargs["data_type"] = f'"{data_type}"'
            value_type = valuetype_from_str(kwargs.get("value_type", None))
            if value_type:
                kwargs["value_type"] = f'"{value_type}"'
                return Scalar(self, "read", [f'"{path}"'], named_input_nodes=kwargs)
            else:
                raise ValueError(
                    "Invalid arguments for reading scalar, value_type must be specified"
                )
        elif data_type == "list":
            # Reading a list have no extra arguments.
            return List(self, "read", [f'"{path}"'])
        else:
            kwargs["data_type"] = '"matrix"'
            self._log.warning(
                "Unknown type read please add a mtd file, or specify in arguments, defaulting to matrix"
            )
            return Matrix(self, "read", [f'"{path}"'], named_input_nodes=kwargs)

    def scalar(self, v: Dict[str, VALID_INPUT_TYPES]) -> Scalar:
        """Construct an scalar value, this can contain str, float, double, integers and booleans.
        :return: A scalar containing the given value.
        """
        if type(v) is str:
            if not ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                v = f'"{v}"'

        # output type assign simply assigns the given variable to the value
        # therefore the output type is assign.
        return Scalar(self, v, assign=True)

    def from_numpy(
        self,
        mat: np.array,
        *args: Sequence[VALID_INPUT_TYPES],
        **kwargs: Dict[str, VALID_INPUT_TYPES],
    ) -> Matrix:
        """Generate DAGNode representing matrix with data given by a numpy array, which will be sent to SystemDS
        on need.

        :param mat: the numpy array
        :param args: unnamed parameters
        :param kwargs: named parameters
        :return: A Matrix
        """

        unnamed_params = ["'./tmp/{file_name}'"]

        if len(mat.shape) == 2:
            named_params = {"rows": mat.shape[0], "cols": mat.shape[1]}
        elif len(mat.shape) == 1:
            named_params = {"rows": mat.shape[0], "cols": 1}
        else:
            # TODO Support tensors.
            raise ValueError("Only two dimensional arrays supported")

        unnamed_params.extend(args)
        named_params.update(kwargs)
        return Matrix(self, "read", unnamed_params, named_params, local_data=mat)

    def from_pandas(
        self,
        df: pd.DataFrame,
        *args: Sequence[VALID_INPUT_TYPES],
        **kwargs: Dict[str, VALID_INPUT_TYPES],
    ) -> Frame:
        """Generate DAGNode representing frame with data given by a pandas dataframe, which will be sent to SystemDS
        on need.

        :param df: the pandas dataframe
        :param args: unnamed parameters
        :param kwargs: named parameters
        :return: A Frame
        """
        unnamed_params = ["'./tmp/{file_name}'"]

        if len(df.shape) == 2:
            named_params = {"rows": df.shape[0], "cols": df.shape[1]}
        elif len(df.shape) == 1:
            named_params = {"rows": df.shape[0], "cols": 1}
        else:
            # TODO Support tensors.
            raise ValueError("Only two dimensional arrays supported")

        unnamed_params.extend(args)
        named_params["data_type"] = '"frame"'

        self._pd_dataframe = df

        named_params.update(kwargs)
        return Frame(self, "read", unnamed_params, named_params, local_data=df)

    def federated(
        self,
        addresses: Iterable[str],
        ranges: Iterable[Tuple[Iterable[int], Iterable[int]]],
        *args,
        **kwargs: Dict[str, VALID_INPUT_TYPES],
    ) -> Matrix:
        """Create federated matrix object.

        :param sds_context: the SystemDS context
        :param addresses: addresses of the federated workers
        :param ranges: for each federated worker a pair of begin and end index of their held matrix
        :param args: unnamed params
        :param kwargs: named params
        :return: The Matrix containing the Federated data.
        """
        addresses_str = "list(" + ",".join(map(lambda s: f'"{s}"', addresses)) + ")"
        ranges_str = "list("
        for begin, end in ranges:
            ranges_str += (
                f'list({",".join(map(str, begin))}), list({",".join(map(str, end))}),'
            )
        ranges_str = ranges_str[:-1]
        ranges_str += ")"
        named_params = {"addresses": addresses_str, "ranges": ranges_str}
        named_params.update(kwargs)
        return Matrix(self, "federated", args, named_params)

    def source(self, path: str, name: str) -> Source:
        """Import methods from a given dml file.

        The importing is done through the DML command source, and adds all defined methods from
        the script to the Source object returned in python. This gives the flexibility to call the methods
        directly on the object returned.

        In systemds a method called func_01 can then be imported using

        ```python
        res = self.sds.source("PATH_TO_FILE", "UNIQUE_NAME").func_01().compute(verbose = True)
        ```

        :param path: The absolute or relative path to the file to import
        :param name: The name to give the imported file in the script, this name must be unique
        """
        return Source(self, path, name)

    def list(
        self, *args: Sequence[VALID_INPUT_TYPES], **kwargs: Dict[str, VALID_INPUT_TYPES]
    ) -> List:
        """Create a List object containing the given nodes.

        Note that only a sequence is allowed, or a dictionary, not both at the same time.
        :param args: A Sequence that will be inserted to a list
        :param kwargs: A Dictionary that will return a dictionary, (internally handled as a list)
        :return: A List
        """
        return List(self, unnamed_input_nodes=args, named_input_nodes=kwargs)

    def combine(self, *args: Sequence[VALID_INPUT_TYPES]) -> Combine:
        """combine nodes to call compute on multiple operations.

        This is usefull for the case of having multiple writes in one script and wanting
        to execute all in one execution reusing intermediates.

        Note this combine does not allow to return anything to the user, so if used,
        please only use nodes that end with either writing or printing elements.

        :param args: A sequence that will be executed with call to compute()
        """
        return Combine(self, unnamed_input_nodes=args)

    def array(self, *args: Sequence[VALID_INPUT_TYPES]) -> List:
        """Create a List object containing the given nodes.

        Note that only a sequence is allowed, or a dictionary, not both at the same time.
        :param args: A Sequence that will be inserted to a list
        :param kwargs: A Dictionary that will return a dictionary, (internally handled as a list)
        :return: A List
        """
        return List(self, unnamed_input_nodes=args)

    def dict(self, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> List:
        """Create a List object containing the given nodes.

        Note that only a sequence is allowed, or a dictionary, not both at the same time.
        :param args: A Sequence that will be inserted to a list
        :param kwargs: A Dictionary that will return a dictionary, (internally handled as a list)
        :return: A List
        """
        return List(self, named_input_nodes=kwargs)

    def __setup_logging(self, level: int, py4j_level: int):
        """Setup the logging infrastructure of the Python API, note this does not effect the JVM part.
        This method also reset the loggers to only have one handler.

        :param level: The SystemDS logging part logging level.
        :param py4j_level: The Py4J logging level.
        """
        # Set py4j level every time
        py4j = logging.getLogger("py4j.java_gateway")
        py4j.setLevel(py4j_level)
        py4j.propagate = False

        if not SystemDSContext._logging_initialized:
            # Add handler only once
            logging.basicConfig()

            root_logger = logging.getLogger(self.__class__.__name__)
            root_logger.handlers.clear()

            f_handler = DynamicStderrHandler()
            f_handler.setLevel(logging.NOTSET)
            f_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    "%y/%m/%d %H:%M:%S",
                )
            )

            root_logger.addHandler(f_handler)
            root_logger.propagate = False

            SystemDSContext._logging_initialized = True

        # Per-instance logger setup
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(level)
        self._log.debug(
            "Logging setup done (SystemDS level: %s, Py4J level: %s)",
            logging.getLevelName(level),
            logging.getLevelName(py4j_level),
        )


class DynamicStderrHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET, formatter=None):
        # Avoid setting stream directly to sys.stderr, we will do that dynamically
        # For more info see test case: tests/basics/test_context_creation/test_random_port_debug3
        # where we redirect the err channel between different contexts
        super().__init__(stream=None)
        self.setLevel(level)
        if formatter:
            self.setFormatter(formatter)

    def emit(self, record):
        # Always use the current sys.stderr when emitting
        self.stream = sys.stderr
        super().emit(record)
