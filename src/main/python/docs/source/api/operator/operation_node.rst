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

Operation Node
==============

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

.. autoclass:: systemds.operator.OperationNode
  :members:

  .. automethod:: __init__
