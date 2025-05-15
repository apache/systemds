.. -------------------------------------------------------------
..
.. Licensed to the Apache Software Foundation (ASF) under one
.. or more contributor license agreements.  See the NOTICE file
.. distributed with this work for additional information
.. regarding copyright ownership.  The ASF licenses this file
.. to you under the Apache License, Version 2.0 (the
.. "License"); you may not use this file except in compliance
.. with the License.  You may obtain a copy of the License at
..
..   http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing,
.. software distributed under the License is distributed on an
.. "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
.. KIND, either express or implied.  See the License for the
.. specific language governing permissions and limitations
.. under the License.
..
.. -------------------------------------------------------------


Install SystemDS
================

SystemDS can be installed via `pip` or installed from source.


Pip
---

Installation is quite simple with `pip`, just execute the following command::

  pip install systemds

SystemDS is a java-project, the `pip` package contains all the necessary `jars`,
but you will need java version 17 installed. Do not use an older or newer
version of java, because SystemDS is non compatible with other java versions.

Check the output of ``java -version``. The output should look similar to::

  openjdk 17.0.11 2024-04-16
  OpenJDK Runtime Environment Temurin-17.0.11+9 (build 17.0.11+9)
  OpenJDK 64-Bit Server VM Temurin-17.0.11+9 (build 17.0.11+9, mixed mode, sharing)

The important part is in the first line ``openjdk version "17.xx"``,
please make sure this is the case.


Source
------

To Install from source involves multiple steps.

Install Dependencies 

- `Maven <https://maven.apache.org/>`_ 
- `Python 3.6+ <https://www.python.org/downloads/>`_ and
- `OpenJDK 17.xxx Java <https://openjdk.java.net/install/>`_

Once installed you please verify your version numbers. 
Additionally you have to install a few python packages.
We suggest to create a new virtual environment using virtual env. 
All commands are run inside src/main/python/.
We assume that in the following scripts python==python3

  python -m venv python_venv 

Now, we activate the environment.

  source python_venv/bin/activate 

In case of using Linux. For Windows PowerShell use:

  Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
  python_venv/Scripts/Activate.ps1 

Note depending on your installation you might need to use pip3 instead of pip::

  pip install numpy py4j wheel requests

Then to build the system you do the following

- Clone the Git Repository: https://github.com/apache/systemds.git
- Open an terminal at the root of the repository.
- Package the Java code using the ``mvn clean package -P distribution`` command
- ``cd src/main/python`` to point at the root of the SystemDS Python library.
- Build the Python API ``python create_python_dist.py``
- Install with ``pip install .``

After this you are ready to go.
