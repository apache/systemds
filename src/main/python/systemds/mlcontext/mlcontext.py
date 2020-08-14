__all__ = ["MLContext"]

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

class MLContext(object):
    """
    Create an MLContext object with spark session or Spark Context
    """

    def __init__(self):
        """
        Starts a new instance of MLContext
        """

        systemds_java_path = os.path.join(get_module_dir(), "systemds-java")
        # nt means its Windows
        cp_separator = ";" if os.name == "nt" else ":"
        lib_cp = os.path.join(systemds_java_path, "lib", "*")
        systemds_cp = os.path.join(systemds_java_path, "*")
        classpath = cp_separator.join([lib_cp, systemds_cp])

        command = ["java", "-cp", classpath]

        sys_root = os.environ.get("SYSTEMDS_ROOT")
        if sys_root != None:
            files = glob(os.path.join(sys_root, "conf", "log4j*.properties"))
            if len(files) > 1:
                print("WARNING: Multiple logging files")
            if len(files) == 0:
                print("WARNING: No log4j file found at: "
                      + os.path.join(sys_root, "conf")
                      + " therefore using default settings")
            else:
                # print("Using Log4J file at " + files[0])
                command.append("-Dlog4j.configuration=file:" + files[0])
        else:
            print("Default Log4J used, since environment $SYSTEMDS_ROOT not set")

        command.append("org.apache.sysds.api.MLContext")

        port = self.__get_open_port()
        command.append(str(port))

        process = Popen(command, stdout=PIPE, stdin=PIPE, stderr=PIPE)

    def close(self):
        """Close the connection to the java process"""
        process : Popen = self.java_gateway.java_process
        self.java_gateway.shutdown()
        os.kill(process.pid, 14)
