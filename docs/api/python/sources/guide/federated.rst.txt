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

.. _Repository: https://github.com/apache/systemml/tree/master/bin/

If that is setup correctly simply start a worker using the following command.
Here the ``8001`` refer to the port used by the worker.::

  systemds WORKER 8001

Simple Aggregation Example
--------------------------

In this example we use a single federated worker, and aggregate the sum of its data.

First we need to create some data for our federated worker to use.
In this example we simply use Numpy to create a ``test.csv`` file::

  # Import numpy
  import numpy as np
  a = np.asarray([[1,2,3], [4,5,6], [7,8,9]])
  np.savetxt("temp/test.csv", a, delimiter=",")

Currently we also require a metadata file for the federated worker.
This should be located next to the ``test.csv`` file called ``test.csv.mtd``.
To make this simply execute the following::

  echo '{ "format":"csv", "header":false, "rows":3, "cols":3 }' > temp/test.csv.mtd

After creating our data we the federated worker becomes able to execute federated instructions.
The aggregated sum using federated instructions in python SystemDS is done as follows::

  # Import numpy and SystemDS federated
  import numpy as np
  from systemds.matrix import Federated
  from systemds.context import SystemDSContext

  # Create a federated matrix
  ## Indicate the dimensions of the data:
  ### Here the first list in the tuple is the top left Coordinate, 
  ### and the second the bottom left coordinate.
  ### It is ordered as [col,row].
  dims = ([0,0], [3,3])

  ## Specify the address + file path from worker:
  address = "localhost:8001/temp/test.csv"

  with SystemDSContext() as sds:
    fed_a = Federated(sds, [address], [dims])
    # Sum the federated matrix and call compute to execute
    print(fed_a.sum().compute())
    # Result should be 45.

Multiple Federated Environments 
-------------------------------

In this example we multiply matrices that are located in different federated environments.

Using the data created from the last example we can simulate
multiple federated workers by starting multiple ones on different ports.
Start with 3 different terminals, and run one federated environment in each.::

  systemds WORKER 8001
  systemds WORKER 8002
  systemds WORKER 8003

Once all three workers are up and running we can leverage all three in the following example::

  # Import numpy and SystemDS federated
  import numpy as np
  from systemds.matrix import Federated
  from systemds.context import SystemDSContext

  addr1 = "localhost:8001/temp/test.csv"
  addr2 = "localhost:8002/temp/test.csv"
  addr3 = "localhost:8003/temp/test.csv"

  # Create a federated matrix using two federated environments
  # Note that the two federated matrices are stacked on top of each other

  with SystemDSContext() as sds:
    fed_a = Federated(sds,
      [addr1, addr2],
      [([0,0], [3,3]), ([0,3], [3,6])])
    
    fed_b = Federated(sds,
      [addr1, addr3],
      [([0,0], [3,3]), ([0,3], [3,6])])
    
    # Multiply, compute and print.
    res = (fed_a * fed_b).compute()

  print(res)

The print should look like::

  [[ 1.  4.  9.  1.  4.  9.]
   [16. 25. 36. 16. 25. 36.]
   [49. 64. 81. 49. 64. 81.]]

.. note::

  If it does not work, then double check 
  that you have:
  
  a csv file, mtd file, and SystemDS Environment is set correctly.
