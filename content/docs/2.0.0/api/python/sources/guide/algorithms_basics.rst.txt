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

This example goes through an algorithm from the list of builtin algorithms that can be applied to a dataset.
For simplicity the dataset used for this is `MNIST <http://yann.lecun.com/exdb/mnist/>`_,
since it is commonly known and explored.

If one wants to skip the explanation then the full script is available at the bottom of this page.

Step 1: Get Dataset
-------------------

SystemDS provides builtin for downloading and setup of the MNIST dataset.
To setup this simply use::

    from systemds.examples.tutorials.mnist import DataManager
    d = DataManager()
    X = d.get_train_data()
    Y = d.get_train_labels()

Here the DataManager contains the code for downloading and setting up numpy arrays containing the data.

Step 2: Reshape & Format
------------------------

Usually data does not come in formats that perfectly fits the algorithms, to make this tutorial more
realistic some data preprocessing is required to change the input to fit.

First the training data, X, has multiple dimensions resulting in a shape (60000, 28, 28).
The dimensions correspond to first the number of images 60000, then the number of row pixels, 28,
and finally the column pixels, 28.

To use this data for logistic regression we have to reduce the dimensions.
The input X is the training data. 
It require the data to have two dimensions, the first resemble the
number of inputs, and the other the number of features.

Therefore to make the data fit the algorithm we reshape the X dataset, like so::

    X = X.reshape((60000, 28*28))

This takes each row of pixels and append to each other making a single feature vector per image.

The Y dataset also does not perfectly fit the logistic regression algorithm, this is because the labels
for this dataset is values ranging from 0, to 9, each label correspond to the integer shown in the image.
unfortunately the algorithm require the labels to be distinct integers from 1 and upwards.

Therefore we add 1 to each label such that the labels go from 1 to 10, like this::

    Y = Y + 1

With these steps we are now ready to train a simple model.

Step 3: Training
----------------

To start with, we setup a SystemDS context::

    from systemds.context import SystemDSContext
    sds = SystemDSContext()

Then setup the data::

    from systemds.matrix import Matrix
    X_ds = Matrix(sds, X)
    Y_ds = Matrix(sds, Y)

to reduce the training time and verify everything works, it is usually good to reduce the amount of data,
to train on a smaller sample to start with::

    sample_size = 1000
    X_ds = Matrix(sds, X[:sample_size])
    Y_ds = Matrix(sds, Y[:sample_size])

And now everything is ready for our algorithm::

    from systemds.operator.algorithm import multiLogReg

    bias = multiLogReg(X_ds, Y_ds)

Note that nothing has been calculated yet, in SystemDS, since it only happens when you call compute::

    bias_r = bias.compute()

bias is a matrix, that if matrix multiplied with an instance returns a value distribution where, the highest value is the predicted type.
This is the matrix that could be saved and used for predicting labels later.

Step 3: Validate
----------------

To see what accuracy the model achieves, we have to load in the test dataset as well.

this can also be extracted from our builtin MNIST loader, to keep the tutorial short the operations are combined::

    Xt = Matrix(sds, d.get_test_data().reshape((10000, 28*28)))
    Yt = Matrix(sds, d.get_test_labels()) + 1

The above loads the test data, and reshapes the X data the same way the training data was reshaped.

Finally we verify the accuracy by calling::

    from systemds.operator.algorithm import multiLogRegPredict
    [m, y_pred, acc] = multiLogRegPredict(Xt, bias, Yt).compute()
    print(acc)

There are three outputs from the multiLogRegPredict call.

- m, is the mean probability of correctly classifying each label.
- y_pred, is the predictions made using the model, bias, trained.
- acc, is the accuracy achieved by the model.

If the subset of the training data is used then you could expect an accuracy of 85% in this example
using 1000 pictures of the training data.

Step 4: Tuning
--------------

Now that we have a working baseline we can start tuning parameters.

But first it is valuable to know how much of a difference in performance there is on the training data, vs the test data.
This gives an indication of if we have exhausted the learning potential of the training data.

To see how our accuracy is on the training data we use the Predict function again, but with our training data::

    [m, y_pred, acc] = multiLogRegPredict(X_ds, bias, Y_ds).compute()
    print(acc)

In this specific case we achieve 100% accuracy on the training data, indicating that we have fit the training data,
and have nothing more to learn from the data as it is now.

To improve further we have to increase the training data, here for example we increase it
from our sample of 1k to the full training dataset of 60k, in this example the maxi is set to reduce the number of iterations the algorithm takes,
to again reduce training time::

    X_ds = Matrix(sds, X)
    Y_ds = Matrix(sds, Y)

    bias = multiLogReg(X_ds, Y_ds, maxi=30)

    [_, _, train_acc] = multiLogRegPredict(X_ds, bias, Y_ds).compute()
    [_, _, test_acc] = multiLogRegPredict(Xt, bias, Yt).compute()
    print(train_acc, "  ", test_acc)

With this change the accuracy achieved changes from the previous value to 92%. This is still low on this dataset as can be seen on `MNIST <http://yann.lecun.com/exdb/mnist/>`_.
But this is a basic implementation that can be replaced by a variety of algorithms and techniques.


Full Script
-----------

The full script, some steps are combined to reduce the overall script. 
One noteworthy change is the + 1 is done on the matrix ready for SystemDS,
this makes SystemDS responsible for adding the 1 to each value.::

    from systemds.context import SystemDSContext
    from systemds.matrix import Matrix
    from systemds.operator.algorithm import multiLogReg, multiLogRegPredict
    from systemds.examples.tutorials.mnist import DataManager

    d = DataManager()

    with SystemDSContext() as sds:
        # Train Data
        X = Matrix(sds, d.get_train_data().reshape((60000, 28*28)))
        Y = Matrix(sds, d.get_train_labels()) + 1.0
        bias = multiLogReg(X, Y, maxi=30)
        # Test data
        Xt = Matrix(sds, d.get_test_data().reshape((10000, 28*28)))
        Yt = Matrix(sds, d.get_test_labels()) + 1.0
        [m, y_pred, acc] = multiLogRegPredict(Xt, bias, Yt).compute()

    print(acc)

