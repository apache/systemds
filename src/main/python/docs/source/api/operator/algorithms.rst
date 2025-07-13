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

Algorithms
==========

SystemDS support different Machine learning algorithms out of the box.

As an example the lm algorithm can be used as follows:

.. code-block:: python

  # Import numpy and SystemDS
  import numpy as np
  from systemds.context import SystemDSContext
  from systemds.operator.algorithm import lm

  # Set a seed
  np.random.seed(0)
  # Generate matrix of feature vectors
  features = np.random.rand(10, 15)
  # Generate a 1-column matrix of response values
  y = np.random.rand(10, 1)

  # compute the weights
  with SystemDSContext() as sds:
    weights = lm(sds.from_numpy(features), sds.from_numpy(y)).compute()
    print(weights)

The output should be similar to

.. code-block:: python

  [[-0.11538199]
  [-0.20386541]
  [-0.39956035]
  [ 1.04078623]
  [ 0.4327084 ]
  [ 0.18954599]
  [ 0.49858968]
  [-0.26812763]
  [ 0.09961844]
  [-0.57000751]
  [-0.43386048]
  [ 0.55358873]
  [-0.54638565]
  [ 0.2205885 ]
  [ 0.37957689]]

.. automodule:: systemds.operator.algorithm
  
.. toctree::
   :maxdepth: 1
   :glob:

   algorithms/*