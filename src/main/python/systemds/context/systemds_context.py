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
            root =  os.path.join(get_module_dir(), "systemds-java")
        
        # nt means its Windows
        cp_separator = ";" if os.name == "nt" else ":"
        lib_cp = os.path.join(root, "target","lib", "*")
        systemds_cp = os.path.join(root,"target","SystemDS.jar")
        classpath = cp_separator.join([lib_cp , systemds_cp])

        command = ["java", "-cp", classpath]

        if os.environ.get("SYSTEMDS_ROOT") != None:
            files = glob(os.path.join(root, "conf", "log4j*.properties"))
            if len(files) > 1:
                print("WARNING: Multiple logging files found selecting: " + files[0])
            if len(files) == 0:
                print("WARNING: No log4j file found at: "
                      + os.path.join(root, "conf")
                      + " therefore using default settings")
            else:
                command.append("-Dlog4j.configuration=file:" + files[0])

        command.append("org.apache.sysds.api.PythonDMLScript")

        # TODO add an argument parser here

        # Find a random port, and hope that no other process
        # steals it while we wait for the JVM to startup
        port = self.__get_open_port()
        command.append(str(port))

        process = Popen(command, stdout=PIPE, stdin=PIPE, stderr=PIPE)
        first_stdout = process.stdout.readline()
        
        if(b"GatewayServer Started" in first_stdout):
            print("Startup success")
        else:
            stderr = process.stderr.readline().decode("utf-8")
            if(len(stderr) > 1):
                raise Exception("Exception in startup of GatewayServer: " + stderr)
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
                    raise Exception("Error in startup of systemDS gateway process: \n gateway StdOut: " + str(outputs) + " \n gateway StdErr" + process.stderr.readline().decode("utf-8") )

        # Handle Std out from the subprocess.
        self.__stdout = Queue()
        self.__stderr = Queue()

        Thread(target=self.__enqueue_output, args=(
            process.stdout, self.__stdout), daemon=True).start()
        Thread(target=self.__enqueue_output, args=(
            process.stderr, self.__stderr), daemon=True).start()

        # Py4j connect to the started process.
        gateway_parameters = GatewayParameters(
            port=port, eager_load=True, read_timeout=5)
        self.java_gateway = JavaGateway(
            gateway_parameters=gateway_parameters, java_process=process)

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
