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

QuickStart Onnx
===============

onnx-systemds is a tool for importing/exporting onnx graphs into/from SystemDS DML scripts.

Prerequisites
---------------
to run onnx-systemds you need to:

- install `onnx <https://github.com/onnx/onnx>`_: `Installation instructions <https://github.com/onnx/onnx#installation>`_
- `set up the environment <https://github.com/apache/systemds/blob/master/bin/README.md>`_

Usage
------
An example call from the ``src/main/python`` directory of systemds::

  python -m systemds.onnx_systemds.convert tests/onnx/test_models/simple_mat_add.onnx


This will generate the dml script ``simple_mat_add.dml`` in the current directory.

Run Tests
---------
Form the ``src/main/python`` directory of systemds:

At first generate the test models::

  python tests/onnx/test_models/model_generate.py

Then you can run the tests::

  python -m unittest tests/onnx/test_simple.py


Converter
---------
It is also possible to invoke the converter from within python.

.. autofunction:: systemds.onnx_systemds.convert.onnx2systemds