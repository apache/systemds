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


Matrix API
==========

SystemDSContext
---------------

All operations using SystemDS need a java instance running.
The connection is ensured by an ``SystemDSContext`` object.
An ``SystemDSContext`` object can be created using:

.. code_block:: python
  sysds = SystemDSContext()

When the calculations are finished the context has to be closed again:

.. code_block:: python
  sysds.close()

Since it is annoying that it is always necessary to close the context, ``SystemDSContext``
implements the python context management protocol, which supports the following syntax:

.. code_block:: python
  with SystemDSContext() as sds:
    # do something with sds which is an SystemDSContext
    pass

This will automatically close the ``SystemDSContext`` once the with-block is left.

.. note::

  Creating a context is an expensive procedure, because a sub-process starting a JVM might have to start, therefore
  try to do this only once for your program, or always leave at least one context open.

.. autoclass:: systemds.context.SystemDSContext
  :members:

OperationNode
-------------

  .. todo
    The explanation for overloade methods seems weird and does not really describe which
    methods we mean (magic methods for operators like `+`, `*` etc.).
    Also I don't understand why that would mean that they return an ``OpeartionNode``.

An ``OperationNode`` represents an operation that executes in SystemDS.
Most methods are overloaded for ``OperationNode``.
This means that they return an ``OperationNode``.
To get the result from an ``OperationNode`` you simply call ``.compute()`` on it, thereby getting the numpy equivalent result.
Even comparisons like ``__eq__``, ``__lt__`` etc. return ``OperationNode``.

.. note::

  All operations are lazily evaluated, meaning before calling ``.compute()`` nothing will be executed in SystemDS.
  Therefore errors will not immediately be recognized while constructing an sequence of operators.

.. autoclass:: systemds.matrix.OperationNode
  :members:

Matrix
------

A ``Matrix`` is represented either by an ``OperationNode``, or the derived class ``Matrix``.
An Matrix can be recognized it by checking the ``output_type`` of the object.

Matrices are the most fundamental objects SystemDS operates on.

Although it is possible to generate matrices with the function calls or object construction specified below,
the recommended way is to use the methods defined on ``SystemDSContext``.

.. autoclass:: systemds.matrix.Matrix
    :members:

.. autofunction:: systemds.matrix.federated

.. autofunction:: systemds.matrix.full

.. autofunction:: systemds.matrix.seq

.. autofunction:: systemds.matrix.rand