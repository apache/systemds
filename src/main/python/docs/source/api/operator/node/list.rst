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

List
====

A ``List`` is represented either by an ``OperationNode``, or the derived class ``List``.

List can contain any of the other types: frame, matrix, scalar and itself list.
The list can be handled like a dictionary or a list primitive, since both access patters are the same at
dml script level.

Although it is possible to generate lists with the function calls or object construction specified below,
the recommended way is to use the methods defined on ``SystemDSContext``, to read in a list from disk,
or construct one using either constructors `array`, `dict` or `list` provided in ``SystemDSContext``.

.. autoclass:: systemds.operator.List
    :members:
    
    .. automethod:: __init__