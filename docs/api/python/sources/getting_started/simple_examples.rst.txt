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

QuickStart
==========

Let's take a look at some code examples.

Matrix Operations
-----------------

Making use of SystemDS, let us multiply an Matrix with an scalar:

.. code-block:: python

  # Import SystemDSContext
  from systemds.context import SystemDSContext
  from systemds.matrix.data_gen import full
  # Create a context and if necessary (no SystemDS py4j instance running)
  # it starts a subprocess which does the execution in SystemDS
  with SystemDSContext() as sds:
      # Full generates a matrix completely filled with one number.
      # Generate a 5x10 matrix filled with 4.2
      m = full(sds,(5, 10), 4.20)
      # multiply with scalar. Nothing is executed yet!
      m_res = m * 3.1
      # Do the calculation in SystemDS by calling compute().
      # The returned value is an numpy array that can be directly printed.
      print(m_res.compute())
  # context will automatically be closed and process stopped

As output we get::

  [[ 13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02]
   [ 13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02]
   [ 13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02]
   [ 13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02]
   [ 13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02  13.02]]

The Python SystemDS package is compatible with numpy arrays.
Let us do a quick element-wise matrix multiplication of numpy arrays with SystemDS.
Remember to first start up a new terminal:

.. code-block:: python

  import numpy as np  # import numpy

  # Import SystemDSContext
  from systemds.context import SystemDSContext
  from systemds.matrix import Matrix

  # create a random array
  m1 = np.array(np.random.randint(100, size=5 * 5) + 1.01, dtype=np.double)
  m1.shape = (5, 5)
  # create another random array
  m2 = np.array(np.random.randint(5, size=5 * 5) + 1, dtype=np.double)
  m2.shape = (5, 5)

  # Create a context
  with SystemDSContext() as sds:
      # element-wise matrix multiplication, note that nothing is executed yet!
      m_res = Matrix(sds, m1) * Matrix(sds, m2)
      # lets do the actual computation in SystemDS! The result is an numpy array
      m_res_np = m_res.compute()
      print(m_res_np)

More complex operations
-----------------------

SystemDS provides algorithm level functions as built-in functions to simplify development.
One example of this is l2SVM, a high level functions for Data-Scientists. Let's take a look at l2svm:

.. code-block:: python

  # Import numpy and SystemDS matrix
  import numpy as np
  from systemds.context import SystemDSContext
  from systemds.matrix import Matrix
  from systemds.operator.algorithm import l2svm

  # Set a seed
  np.random.seed(0)
  # Generate random features and labels in numpy
  # This can easily be exchanged with a data set.
  features = np.array(np.random.randint(100, size=10 * 10) + 1.01, dtype=np.double)
  features.shape = (10, 10)
  labels = np.zeros((10, 1))

  # l2svm labels can only be 0 or 1
  for i in range(10):
      if np.random.random() > 0.5:
          labels[i][0] = 1

  # compute our model
  with SystemDSContext() as sds:
      model = l2svm(Matrix(sds, features), Matrix(sds, labels)).compute()
      print(model)

The output should be similar to::

  [[ 0.02033445]
   [-0.00324092]
   [ 0.0014692 ]
   [ 0.02649209]
   [-0.00616902]
   [-0.0095087 ]
   [ 0.01039221]
   [-0.0011352 ]
   [-0.01686351]
   [-0.03839821]]

