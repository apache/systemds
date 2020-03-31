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

Since we always need a java instance running which will can execute operations in SystemDS, we
need to start this connection at some point. We do this with ``SystemDSContext``. A ``SystemDSContext``
object has to be created and once we are finished ``.close()`` has to be called on it, or
we can use it by doing ``with SystemDSContext() as context:``, which will automatically close
the context if an error occurs or we are finished with our operations. Creating an context is
an expensive procedure, because we might have to start a subprocess running java, therefore
try to do this only once for your program, or always leave at least one context open.

Our SystemDS operations always start with an call on a ``SystemDSContext``, most likely to generate
a matrix on which we can operate.

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

Matrices are the most fundamental objects we operate on.

Although we can generate matrices with the function calls or object construction specified below,
the recommended way is to use the methods defined on ``SystemDSContext``.

If we can generate the matrix in SystemDS directly via a function call,
an python function exists which will generate an ``OperationNode`` e.g. ``federated``, ``full``, ``seq``,
representing the SystemDS operation.

If we want to work on an numpy array or want to read a matrix from a file, we need to use the class ``Matrix``.

.. autoclass:: systemds.matrix.Matrix
    :members:

.. autofunction:: systemds.matrix.federated

.. autofunction:: systemds.matrix.full

.. autofunction:: systemds.matrix.seq

