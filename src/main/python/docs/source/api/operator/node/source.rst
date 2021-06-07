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

Source
======

A ``Source`` is the action of importing method declarations from other DML scripts.
This function allows one to define a function in DML and use it in the python API.

Although it is possible to generate sources with the function calls or object construction specified below,
the recommended way is to use the method defined on ``SystemDSContext`` called source to construct one
using a path to the dml file to source.

.. autoclass:: systemds.operator.Source
    :members:
    
    .. automethod:: __init__