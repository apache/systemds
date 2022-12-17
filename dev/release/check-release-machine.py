#!/usr/bin/env python3
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

################################################################################
##  File:  check-release-machine.py
##  Desc:  Verify the required software (Maven, gpg, OS type etc.) on the release machine
################################################################################

# https://docs.python.org/3/library/shlex.html
import shlex
from enum import Enum, auto
import os
import sys
import subprocess


class System(Enum):
    Ubuntu = auto()

SUPPORTED_SYSTEMS = {
    System.Ubuntu: {"18", "20"}
}

def check_python_version():

    if sys.version_info.major == 3 and sys.version_info.minor in (6, 7, 8, 9, 10):
        return
    version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    raise RuntimeError("Unsupported Python version {}. ".format(version))

def run(command: str, check=True) -> subprocess.CompletedProcess:

    proc = subprocess.run(shlex.split(command), check=check)

    return proc

def detect_linux_distro() -> (System, str):

    with open('/etc/os-release') as os_release:
        lines = [line.strip() for line in os_release.readlines() if line.strip() != '']
        info = {k: v.strip("'\"") for k, v in (line.split('=', maxsplit=1) for line in lines)}
    
    name = info['NAME']

    if name.startswith("Ubuntu"):
        system = System.Ubuntu
        version = info['VERSION_ID']
    else:
        raise RuntimeError("this os is not supported by this script")
    
    return system, version

def check_linux_distro(sytem: System, version: str) -> bool:
    
    if '.' in version:
        version = version.split('.')[0]
    
    if len(SUPPORTED_SYSTEMS[system]) == 0:
        # print that this script does not support this distro
        return False

    return True

def check_maven_installed() -> bool:

    process = run("which mvn", check=False)
    return process.returncode == 0

def check_gpg_installed() -> bool:

    process = run("whereis gpg", check=False)
    return process.returncode == 0

def main():

    if check_maven_installed():
        print('Maven is available')
    
    if check_gpg_installed():
        print('Gpg is installed')
    
    check_python_version()
    
    system, version = detect_linux_distro()


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print('Failed with exception')
        raise err
