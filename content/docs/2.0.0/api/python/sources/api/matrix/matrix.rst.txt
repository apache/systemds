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

Matrix
======

A ``Matrix`` is represented either by an ``OperationNode``, or the derived class ``Matrix``.
An Matrix can be recognized it by checking the ``output_type`` of the object.

Matrices are the most fundamental objects SystemDS operates on.

Although it is possible to generate matrices with the function calls or object construction specified below,
the recommended way is to use the methods defined on ``SystemDSContext``.

.. autoclass:: systemds.matrix.Matrix
    :members:
    
    .. automethod:: __init__

