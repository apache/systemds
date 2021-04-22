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
but you will need java version 8 installed. Do not use an older or newer
version of java, because SystemDS is non compatible with other java versions.

Check the output of ``java -version``. The output should look similar to::

  openjdk version "1.8.0_242"
  OpenJDK Runtime Environment (build 1.8.0_242-b08)
  OpenJDK 64-Bit Server VM (build 25.242-b08, mixed mode)

The important part is in the first line ``openjdk version "1.8.0_xxx"``,
please make sure this is the case.


Source
------

To Install from source involves three steps.

Install Dependencies 

- `Maven <https://maven.apache.org/>`_ 
- `Python 3.6+ <https://www.python.org/downloads/>`_ and
- `OpenJDK 1.8 Java <https://openjdk.java.net/install/>`_

Once installed you please verify your version numbers. 
Additionally you have to install a few python packages.
Note depending on your installation you might need to use pip3 instead of pip::

  pip install numpy py4j wheel jinja2 onnx requests

Then to build the system you do the following

- Clone the Git Repository: https://github.com/apache/systemds.git
- Open an terminal at the root of the repository.
- Package the Java code using the ``mvn clean package -P distribution`` command
- ``cd src/main/python`` to point at the root of the SystemDS Python library.
- Copy `jars` with ``python pre_setup.py``
- Install with ``pip install .``

After this you are ready to go.
