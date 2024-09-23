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

.. include:: ../code/getting_started/simpleExamples/multiply.py
  :code: python
  :start-line: 20
  :encoding: utf-8
  :literal:

As output we get

.. code-block::

  [[13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02]
   [13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02]
   [13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02]
   [13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02]
   [13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02 13.02]]

The Python SystemDS package is compatible with numpy arrays.
Let us do a quick element-wise matrix multiplication of numpy arrays with SystemDS.
Remember to first start up a new terminal:

.. include:: ../code/getting_started/simpleExamples/multiplyMatrix.py
  :code: python
  :start-line: 20
  :encoding: utf-8
  :literal:

More complex operations
-----------------------

SystemDS provides algorithm level functions as built-in functions to simplify development.
One example of this is l2SVM, a high level functions for Data-Scientists. Let's take a look at l2svm:

.. include:: ../code/getting_started/simpleExamples/l2svm.py
  :code: python
  :start-line: 20
  :encoding: utf-8
  :literal:

The output should be similar to

.. code-block::

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

To get the full performance of SystemDS one can modify the script to only use internal functionality,
instead of using numpy arrays that have to be transfered into systemDS.
The above script transformed goes like this:

.. include:: ../code/getting_started/simpleExamples/l2svm_internal.py
  :code: python
  :start-line: 20
  :encoding: utf-8
  :literal:

When reading in datasets for processing it is highly recommended that you read from inside systemds using 
sds.read("file"), since this avoid the transferring of numpy arrays.
