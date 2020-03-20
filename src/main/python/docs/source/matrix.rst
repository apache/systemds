.. ------------------------------------------------------------------------------
..  Copyright 2020 Graz University of Technology
..
..  Licensed under the Apache License, Version 2.0 (the "License");
..  you may not use this file except in compliance with the License.
..  You may obtain a copy of the License at
..
..    http://www.apache.org/licenses/LICENSE-2.0
..
..  Unless required by applicable law or agreed to in writing, software
..  distributed under the License is distributed on an "AS IS" BASIS,
..  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
..  See the License for the specific language governing permissions and
..  limitations under the License.
.. ------------------------------------------------------------------------------

Matrix API
==========

OperationNode
-------------

An ``OperationNode`` represents an operation that executes in SystemDS.
Most methods are overloaded for ``OperationNode``.
This means that they return an ``OperationNode``.
To get the result from an `OperationNode` you simply call ``.compute()`` on it, thereby getting the numpy equivalent result.
Even comparisons like ``__eq__``, ``__lt__`` etc. gives `OperationNode`s.

.. note::

  All operations are lazily evaluated, meaning before calling ``.compute()`` nothing will be executed in SystemDS.
  Therefore errors will not immediately be recognized while constructing an sequence of operators.

.. autoclass:: systemds.matrix.OperationNode
  :members:

Matrix
------

A `Matrix` is represented either by an `OperationNode`, or the derived class `Matrix`.
An Matrix can recognized it by checking the ``output_type`` of the object.

Matrices are the most fundamental objects we operate on.
If one generate the matrix in SystemDS directly via a function call,
it can be used in an function which will generate an `OperationNode` e.g. `federated`, `full`, `seq`.

If we want to work on an numpy array we need to use the class `Matrix`.

.. autoclass:: systemds.matrix.Matrix
    :members:

.. autofunction:: systemds.matrix.federated

.. autofunction:: systemds.matrix.full

.. autofunction:: systemds.matrix.seq

