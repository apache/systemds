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
.. ------------------------------------------------------------

Federated Environment
=====================

The python SystemDS supports federated execution.
To enable this, each of the federated environments have to have 
a running federated worker.

Start Federated worker
----------------------

To start a federated worker, you first have to setup your environment variables.
A simple guide to do this is in the SystemDS Repository_.

.. _Repository: https://github.com/apache/systemds/tree/main/bin/

If that is setup correctly simply start a worker using the following command.
Here the ``8001`` refer to the port used by the worker.

.. code-block::

  systemds WORKER 8001

Simple Aggregation Example
--------------------------

In this example we use a single federated worker, and aggregate the sum of its data.

First we need to create some data for our federated worker to use.
In this example we simply use Numpy to create a ``test.csv`` file.

Currently we also require a metadata file for the federated worker.
This should be located next to the ``test.csv`` file called ``test.csv.mtd``.
To make both the data and metadata simply execute the following

.. include:: ../code/guide/federated/federatedTutorial_part1.py
  :code: python
  :start-line: 20
  :encoding: utf-8
  :literal:

After creating our data the federated worker becomes able to execute federated instructions.
The aggregated sum using federated instructions in python SystemDS is done as follows

.. include:: ../code/guide/federated/federatedTutorial_part2.py
  :code: python
  :start-line: 20
  :encoding: utf-8
  :literal:

Multiple Federated Environments 
-------------------------------

In this example we multiply matrices that are located in different federated environments.

Using the data created from the last example we can simulate
multiple federated workers by starting multiple ones on different ports.
Start with 3 different terminals, and run one federated environment in each.

.. code-block::

  systemds WORKER 8001
  systemds WORKER 8002
  systemds WORKER 8003

Once all three workers are up and running we can leverage all three in the following example

.. include:: ../code/guide/federated/federatedTutorial_part3.py
  :code: python
  :start-line: 20
  :encoding: utf-8
  :literal:

The print should look like

.. code-block::

  [[198. 243. 288.]
   [198. 243. 288.]
   [198. 243. 288.]]

.. note::

  If it does not work, then double check 
  that you have:
  
  a csv file, mtd file, and SystemDS Environment is set correctly.

Multi-tenant Federated Learning
-------------------------------

SystemDS supports Multi-tenant Federated Learning, meaning that multiple
coordinators learn on shared federated workers. From another perspective,
the federated worker allows multiple coordinators to perform model training
simultaneously using the data from the respective federated site. This
approach enables the worker to operate in a server-like mode, providing
multiple tenants with the ability to learn on the federated data at the same
time. Tenant isolation ensures that tenant-specific intermediate results are
only accessible by the respective tenant.

Limitations
~~~~~~~~~~~

Since the coordinators are differentiated by their IP address in combination
with their process ID, the worker is not able to isolate coordinators which
share the same IP address and the same process ID. This occurs, for example,
when two coordinators are running behind a proxy (same IP address), where
both coordinators coincidentally have the same process ID.

A second limitation is showing up in networks using the Dynamic Host Protocol
(DHCP). Since the federated worker identifies the coordinator based on the
IP address, the worker does not re-identify the coordinator when its IP address
has changed, i.e., when DHCP renews its IP address.
