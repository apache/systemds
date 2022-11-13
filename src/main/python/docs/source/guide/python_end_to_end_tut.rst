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

Python end-to-end tutorial
==========================

The goal of this tutorial is to showcase different features of the SystemDS framework that can be accessed with the Python API.
For this, we want to use the `Adult <https://archive.ics.uci.edu/ml/datasets/adult/>`_ dataset and predict whether the income of a person exceeds $50K/yr based on census data.
The Adult dataset contains attributes like age, workclass, education, marital-status, occupation, race, [...] and the labels >50K or <=50K.
Most of these features are categorical string values, but the dataset also includes continuous features.
For this, we define three different levels with an increasing level of detail with regard to features provided by SystemDS.
In the first level, shows the built-in preprocessing capabilities of SystemDS.
With the second level, we want to show how we can integrate custom-built networks or algorithms into our Python program.

Prerequisite: 

- :doc:`/getting_started/install`

Level 1
-------

This example shows how one can work the SystemDS framework.
More precisely, we will make use of the built-in DataManager, Multinomial Logistic Regression function, and the Confusion Matrix function.
The dataset used in this tutorial is a preprocessed version of the "UCI Adult Data Set".
If one wants to skip the explanation then the full script is available at the end of this level.

We will train a Multinomial Logistic Regression model on the training dataset and subsequently use the test dataset
to assess how well our model can predict if the income is above or below $50K/yr based on the features.

Step 1: Load and prepare data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we get our training and testing data from the built-in DataManager. Since the multiLogReg function requires the
labels (Y) to be > 0, we add 1 to all labels. This ensures that the smallest label is >= 1. Additionally we will only take
a fraction of the training and test set into account to speed up the execution.

.. include:: ../code/guide/end_to_end/part1.py
  :code: python
  :start-line: 20
  :end-line: 51

Here the DataManager contains the code for downloading and setting up either Pandas DataFrames or internal SystemDS Frames,
for the best performance and no data transfer from pandas to SystemDS it is recommended to read directly from disk into SystemDS.

Step 2: Training
~~~~~~~~~~~~~~~~

Now that we prepared the data, we can use the multiLogReg function. First, we will train the model on our
training data. Afterward, we can make predictions on the test data and assess the performance of the model.

.. include:: ../code/guide/end_to_end/part1.py
  :code: python
  :start-line: 53
  :end-line: 54

Note that nothing has been calculated yet. In SystemDS the calculation is executed once compute() is called.
E.g. betas_res = betas.compute().

We can now use the trained model to make predictions on the test data.

.. include:: ../code/guide/end_to_end/part1.py
  :code: python
  :start-line: 56
  :end-line: 57

The multiLogRegPredict function has three return values:
    - m, a matrix with the mean probability of correctly classifying each label. We do not use it further in this example.
    - y_pred, is the predictions made using the model
    - acc, is the accuracy achieved by the model.

Step 3: Confusion Matrix
~~~~~~~~~~~~~~~~~~~~~~~~

A confusion matrix is a useful tool to analyze the performance of the model and to obtain a better understanding
which classes the model has difficulties separating.
The confusionMatrix function takes the predicted labels and the true labels. It then returns the confusion matrix
for the predictions and the confusion matrix averages of each true class.

.. include:: ../code/guide/end_to_end/part1.py
  :code: python
  :start-line: 59
  :end-line: 60

Full Script
~~~~~~~~~~~

In the full script, some steps are combined to reduce the overall script.

.. include:: ../code/guide/end_to_end/part1.py
  :code: python
  :start-line: 20
  :end-line: 65

Level 2
-------

In this level we want to show how we can integrate a custom built algorithm using the Python API.
For this we will introduce another dml file, which can be used to train a basic feed forward network.

Step 1: Obtain data
~~~~~~~~~~~~~~~~~~~

For the whole data setup please refer to level 1, Step 1, as these steps are identical.

.. include:: ../code/guide/end_to_end/part2.py
  :code: python
  :start-line: 20
  :end-line: 51

Step 2: Load the algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~

We use a neural network with 2 hidden layers, each consisting of 200 neurons.
First, we need to source the dml file for neural networks.
This file includes all the necessary functions for training, evaluating, and storing the model.
The returned object of the source call is further used for calling the functions.
The file can be found here:

    - :doc:tests/examples/tutorials/neural_net_source.dml

.. include:: ../code/guide/end_to_end/part2.py
  :code: python
  :start-line: 54
  :end-line: 55


Step 3: Training the neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training a neural network in SystemDS using the train function is straightforward.
The first two arguments are the training features and the target values we want to fit our model on.
Then we need to set the hyperparameters of the model.
We choose to train for 1 epoch with a batch size of 16 and a learning rate of 0.01, which are common parameters for neural networks.
The seed argument ensures that running the code again yields the same results.

.. include:: ../code/guide/end_to_end/part2.py
  :code: python
  :start-line: 61
  :end-line: 62


Step 4: Saving the model
~~~~~~~~~~~~~~~~~~~~~~~~

For later usage, we can save the trained model.
We only need to specify the name of our model and the file path.
This call stores the weights and biases of our model.

.. include:: ../code/guide/end_to_end/part2.py
  :code: python
  :start-line: 64
  :end-line: 65


Full Script NN
~~~~~~~~~~~---

The complete script now can be seen here:


.. include:: ../code/guide/end_to_end/part2.py
  :code: python
  :start-line: 20
  :end-line: 64
