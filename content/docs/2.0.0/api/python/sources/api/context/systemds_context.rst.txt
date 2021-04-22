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

SystemDSContext
===============

All operations using SystemDS need a java instance running.
The connection is ensured by an ``SystemDSContext`` object.
An ``SystemDSContext`` object can be created using::

  from systemds.context import SystemDSContext
  sds = SystemDSContext()

When the calculations are finished the context has to be closed again::

  sds.close()

Since it is annoying that it is always necessary to close the context, ``SystemDSContext``
implements the python context management protocol, which supports the following syntax::

  with SystemDSContext() as sds:
    # do something with sds which is an SystemDSContext
    pass

This will automatically close the ``SystemDSContext`` once the with-block is left.

.. note::

  Creating a context is an expensive procedure, because a sub-process starting a JVM might have to start, therefore
  try to do this only once for your program, or always leave at least one context open.

.. autoclass:: systemds.context.SystemDSContext
  :members:

  .. automethod:: __init__