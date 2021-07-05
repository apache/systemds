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

Built-in Algorithms 
===================

Prerequisite: 

- :doc:`/getting_started/install`

This example shows how one can work with Numpy data within the SystemDs framework.
More precisely we will make use of the build in DataManager, Multinomial Logistic Regression function and the
confusion Matrix function.
The dataset used in this tutorial is a preprocessed version
of the "UCI Adult Data Set". If you are interested in data preprocessing there is a separate tutorial on this topic.
If one wants to skip the explanation then the full script is available at the bottom of this page.

Our goal will be to predict whether the income of a person exceeds $50K/yr based on census data. The Adult
Dataset contains attributes like: age, workclass, education, marital-status, occupation, race, [...] and the labels
>50K or <=50K. We will train a Multinomial Logistic Regression model on the training dataset and subsequently we
will use the test dataset to asses how well our model can predict if the income is above or below $50K/yr based on the
attributes.

Step 1: Load and prepare data
-------------------

First we get our training and testing data from the build in DataManager. Since the multiLogReg function the
labels (Y) to be > 0 we add 1 to all labels. This ensures that the smallest label is >= 1. Addtionally we will only take
a fraction of the training and test set into account to speed up the execution.

.. code-block:: python

    from systemds.examples.tutorials.mnist import DataManager
    d = DataManager()
    X = d.get_preprocessed_dataset()
    Y = d.get_preprocessed_dataset()

    # limit the sample size
    train_count = 15000
    test_count = 5000
    # Train data
    X = self.sds.from_numpy(train_data[:train_count])
    Y = self.sds.from_numpy(train_labels[:train_count])
    Y = Y + 1.0

    # Test data
    Xt = self.sds.from_numpy(test_data[:test_count])
    Yt = self.sds.from_numpy(test_labels[:test_count])
    Yt = Yt + 1.0

Here the DataManager contains the code for downloading and setting up numpy arrays containing the data.

Step 2: Training
------------------------

Now that we prepared the data we can use the multiLogReg function. First we will train the model on our
training data. Afterwards we can make predictions on the test data and asses the performance of the model.

.. code-block:: python

    from systemds.operator.algorithm import multiLogReg
    betas = multiLogReg(X, Y)

Note that nothing has been calculated yet. In SystemDS the calculation is executed once .compute() is called.
E.g. betas_res = betas.compute().


We can now use the trained model to make predictions on the test data.

.. code-block:: python

    [_, y_pred, acc] = multiLogRegPredict(Xt, betas, Yt).compute()
    self.assertGreater(acc, 80)


The multiLogRegPredict function has three return values:
    - m, a matrix with the mean probability of correctly classifying each label. We not use it further in this example.
    - y_pred, is the predictions made using the model
    - acc, is the accuracy achieved by the model.

Step 3: Confusion Matrix
----------------

A confusion matrix is a useful tool to analyze the performance of the model and to obtain a better understanding
which classes the model has difficulties to separate.
The confusionMatrix function takes the predicted labels and the true labels. It then returns the confusion matrix
for the predictions and the confusion matrix averages of each true class.

If you followed the tutorial you should be able to verify the results with the provided assertTrue function call.

.. code-block:: python

    confusion_matrix_abs, _ = confusionMatrix(self.sds.from_numpy(y_pred), Yt).compute()
    self.assertTrue(
        np.allclose(
            confusion_matrix_abs,
            np.array([[3583, 502],
                      [245, 670]])
        )
    )



Full Script
-----------

The full script, some steps are combined to reduce the overall script.

.. code-block:: python

    from systemds.examples.tutorials.mnist import DataManager
    from systemds.operator.algorithm import multiLogReg

    d = DataManager()
    X = d.get_preprocessed_dataset()
    Y = d.get_preprocessed_dataset()

    # limit the sample size
    train_count = 15000
    test_count = 5000
    # Train data
    X = self.sds.from_numpy(train_data[:train_count])
    Y = self.sds.from_numpy(train_labels[:train_count])
    Y = Y + 1.0

    # Test data
    Xt = self.sds.from_numpy(test_data[:test_count])
    Yt = self.sds.from_numpy(test_labels[:test_count])
    Yt = Yt + 1.0

    betas = multiLogReg(X, Y)
    [_, y_pred, acc] = multiLogRegPredict(Xt, betas, Yt).compute()

    confusion_matrix_abs, _ = confusionMatrix(self.sds.from_numpy(y_pred), Yt).compute()
    self.assertTrue(
        np.allclose(
            confusion_matrix_abs,
            np.array([[3583, 502],
                      [245, 670]])
        )
    )



