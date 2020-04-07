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

# Introduction

Keras2DML converts a Keras specification to DML through the intermediate Caffe2DML module. 
It is designed to fit well into the mllearn framework and hence supports NumPy, Pandas as well as PySpark DataFrame.

First, install SystemDS and other dependencies for the below demo:

```
pip install systemml keras tensorflow
``` 

To create a Keras2DML object, simply pass the keras object to the Keras2DML constructor. It's also important to note that your models
should be compiled so that the loss can be accessed for Caffe2DML.

# Training Lenet on the MNIST dataset

Download the MNIST dataset using [mlxtend package](https://pypi.python.org/pypi/mlxtend).

```python
# pyspark --driver-memory 20g

# Disable Tensorflow from using GPU to avoid unnecessary evictions by SystemDS runtime
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

# Train Lenet using SystemDS
from systemml.mllearn import Keras2DML
sysml_model = Keras2DML(spark, keras_model, weights='weights_dir')
# sysml_model.setConfigProperty("sysml.native.blas", "auto")
# sysml_model.setGPU(True).setForceGPU(True)
sysml_model.fit(X_train, y_train)
sysml_model.score(X_test, y_test)
```

# Prediction using a pretrained ResNet-50

```python
# pyspark --driver-memory 20g
# Disable Tensorflow from using GPU to avoid unnecessary evictions by SystemDS runtime
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Set channel first layer
from keras import backend as K
K.set_image_data_format('channels_first')

from systemml.mllearn import Keras2DML
import systemml as sml
import keras, urllib
from PIL import Image
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50

keras_model = ResNet50(weights='imagenet',include_top=True,pooling='None',input_shape=(3,224,224))
keras_model.compile(optimizer='sgd', loss= 'categorical_crossentropy')

sysml_model = Keras2DML(spark,keras_model,input_shape=(3,224,224), weights='weights_dir', labels='https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/imagenet/labels.txt')
sysml_model.summary()
urllib.urlretrieve('https://upload.wikimedia.org/wikipedia/commons/f/f4/Cougar_sitting.jpg', 'test.jpg')
img_shape = (3, 224, 224)
input_image = sml.convertImageToNumPyArr(Image.open('test.jpg'), img_shape=img_shape)
sysml_model.predict(input_image)
```

Please see [Keras2DML's reference guide](http://apache.github.io/systemml/reference-guide-keras2dml) for more details.
