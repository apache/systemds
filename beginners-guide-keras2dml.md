---
layout: global
title: Beginner's Guide for Keras2DML users
description: Beginner's Guide for Keras2DML users
---
<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

* This will become a table of contents (this text will be scraped).
{:toc}

<br/>

## Introduction

Keras2DML is an **experimental API** that converts a Keras specification to DML through the intermediate Caffe2DML module. 
It is designed to fit well into the mllearn framework and hence supports NumPy, Pandas as well as PySpark DataFrame.

### Getting Started 

To create a Keras2DML object, one needs to create a Keras model through the Funcitonal API. please see the [Functional API.](https://keras.io/models/model/)
This module utilizes the existing [Caffe2DML](beginners-guide-caffe2dml) backend to convert Keras models into DML. Keras models are 
parsed and translated into Caffe prototext and caffemodel files which are then piped into Caffe2DML. Thus one can follow the Caffe2DML
documentation for further information.

### Model Conversion

Keras models are parsed based on their layer structure and corresponding weights and translated into the relative Caffe layer and weight
configuration. Be aware that currently this is a translation into Caffe and there will be loss of information from keras models such as 
intializer information, and other layers which do not exist in Caffe. 

First, install SystemML and other dependencies for the below demo:

```
pip install systemml keras tensorflow mlxtend
``` 

To create a Keras2DML object, simply pass the keras object to the Keras2DML constructor. It's also important to note that your models
should be compiled so that the loss can be accessed for Caffe2DML.



```python
# pyspark --driver-memory 20g

# Disable Tensorflow from using GPU to avoid unnecessary evictions by SystemML runtime
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import dependencies
from mlxtend.data import mnist_data
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout,Flatten
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD

# Set channel first layer
K.set_image_data_format('channels_first')

# Download the MNIST dataset
X, y = mnist_data()
X, y = shuffle(X, y)

# Split the data into training and test
n_samples = len(X)
X_train = X[:int(.9 * n_samples)]
y_train = y[:int(.9 * n_samples)]
X_test = X[int(.9 * n_samples):]
y_test = y[int(.9 * n_samples):]

# Define Lenet in Keras
keras_model = Sequential()
keras_model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(1,28,28), padding='same'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Flatten())
keras_model.add(Dense(512, activation='relu'))
keras_model.add(Dropout(0.5))
keras_model.add(Dense(10, activation='softmax'))
keras_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))
keras_model.summary()

# Scale the input features
scale = 0.00390625
X_train = X_train*scale
X_test = X_test*scale

# Train Lenet using SystemML
from systemml.mllearn import Keras2DML
sysml_model = Keras2DML(spark, keras_model, weights='weights_dir')
# sysml_model.setConfigProperty("sysml.native.blas", "auto")
# sysml_model.setGPU(True).setForceGPU(True)
sysml_model.fit(X_train, y_train)
sysml_model.score(X_test, y_test)
```

# Frequently asked questions

#### How can I get the training and prediction DML script for the Keras model?

The training and prediction DML scripts can be generated using `get_training_script()` and `get_prediction_script()` methods.

```python
from systemml.mllearn import Keras2DML
sysml_model = Keras2DML(spark, keras_model, input_shape=(3,224,224))
print(sysml_model.get_training_script())
```

#### What is the mapping between Keras' parameters and Caffe's solver specification ? 

|                                                        | Specified via the given parameter in the Keras2DML constructor | From input Keras' model                                                                 | Corresponding parameter in the Caffe solver file |
|--------------------------------------------------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------------------------------------|
| Solver type                                            |                                                                | `type(keras_model.optimizer)`. Supported types: `keras.optimizers.{SGD, Adagrad, Adam}` | `type`                                           |
| Maximum number of iterations                           | `max_iter`                                                     | The `epoch` parameter in the `fit` method is not supported.                             | `max_iter`                                       |
| Validation dataset                                     | `test_iter` (explained in the below section)                   | The `validation_data` parameter in the `fit` method is not supported.                   | `test_iter`                                      |
| Monitoring the loss                                    | `display, test_interval` (explained in the below section)      | The `LossHistory` callback in the `fit` method is not supported.                        | `display, test_interval`                         |
| Learning rate schedule                                 | `lr_policy`                                                    | The `LearningRateScheduler` callback in the `fit` method is not supported.              | `lr_policy` (default: step)                      |
| Base learning rate                                     |                                                                | `keras_model.optimizer.lr`                                                              | `base_lr`                                        |
| Learning rate decay over each update                   |                                                                | `keras_model.optimizer.decay`                                                           | `gamma`                                          |
| Global regularizer to use for all layers               | `regularization_type,weight_decay`                             | The current version of Keras2DML doesnot support custom regularizers per layer.         | `regularization_type,weight_decay`               |
| If type of the optimizer is `keras.optimizers.SGD`     |                                                                | `momentum, nesterov`                                                                    | `momentum, type`                                 |
| If type of the optimizer is `keras.optimizers.Adam`    |                                                                | `beta_1, beta_2, epsilon`. The parameter `amsgrad` is not supported.                    | `momentum, momentum2, delta`                     |
| If type of the optimizer is `keras.optimizers.Adagrad` |                                                                | `epsilon`                                                                               | `delta`                                          |

#### How do I specify the batch size and the number of epochs ?

Since Keras2DML is a mllearn API, it doesnot accept the batch size and number of epochs as the parameter in the `fit` method.
Instead, these parameters are passed via `batch_size` and `max_iter` parameters in the Keras2DML constructor.
For example, the equivalent Python code for `keras_model.fit(features, labels, epochs=10, batch_size=64)` is as follows:

```python
from systemml.mllearn import Keras2DML
epochs = 10
batch_size = 64
num_samples = features.shape[0]
max_iter = int(epochs*math.ceil(num_samples/batch_size))
sysml_model = Keras2DML(spark, keras_model, batch_size=batch_size, max_iter=max_iter, ...)
sysml_model.fit(features, labels)
``` 

#### What optimizer and loss does Keras2DML use by default if `keras_model` is not compiled ?

If the user does not `compile` the keras model, then we throw an error.

For classification applications, you can consider using cross entropy loss and SGD optimizer with nesterov momentum:

```python 
keras_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True))
```

Please refer to [Keras's documentation](https://keras.io/losses/) for more detail.

#### What is the learning rate schedule used ?

Keras2DML does not support the `LearningRateScheduler` callback. 
Instead one can set the custom learning rate schedule to one of the following schedules by using the `lr_policy` parameter of the constructor:
- `step`: return `base_lr * gamma ^ (floor(iter / step))` (default schedule)
- `fixed`: always return `base_lr`.
- `exp`: return `base_lr * gamma ^ iter`
- `inv`: return `base_lr * (1 + gamma * iter) ^ (- power)`
- `poly`: the effective learning rate follows a polynomial decay, to be zero by the max_iter. return `base_lr (1 - iter/max_iter) ^ (power)`
- `sigmoid`: the effective learning rate follows a sigmod decay return b`ase_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))`

#### How to set the size of the validation dataset ?

The size of the validation dataset is determined by the parameters `test_iter` and the batch size. For example: If the batch size is 64 and 
`test_iter` is set to 10 in the `Keras2DML`'s constructor, then the validation size is 640. This setting generates following DML code internally:

```python
num_images = nrow(y_full)
BATCH_SIZE = 64
num_validation = 10 * BATCH_SIZE
X = X_full[(num_validation+1):num_images,]; y = y_full[(num_validation+1):num_images,]
X_val = X_full[1:num_validation,]; y_val = y_full[1:num_validation,]
num_images = nrow(y)
``` 

#### How to monitor loss via command-line ?

To monitor loss, please set the parameters `display`, `test_iter` and `test_interval` in the `Keras2DML`'s constructor.  
For example: for the expression `Keras2DML(..., display=100, test_iter=10, test_interval=500)`, we
- display the training loss and accuracy every 100 iterations and
- carry out validation every 500 training iterations and display validation loss and accuracy.

#### How do you ensure that Keras2DML produce same results as other Keras' backend?

To verify that Keras2DML produce same results as other Keras' backend, we have [Python unit tests](https://github.com/apache/systemml/blob/master/src/main/python/tests/test_nn_numpy.py)
that compare the results of Keras2DML with that of TensorFlow. We assume that Keras team ensure that all their backends are consistent with their TensorFlow backend.


