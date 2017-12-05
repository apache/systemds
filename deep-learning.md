---
layout: global
title: Deep Learning with SystemML
description: Deep Learning with SystemML
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

There are three different ways to implement a Deep Learning model in SystemML:
1. Using the [DML-bodied NN library](https://github.com/apache/systemml/tree/master/scripts/nn): This library allows the user to exploit full flexibility of [DML language](http://apache.github.io/systemml/dml-language-reference) to implement your neural network.
2. Using the experimental [Caffe2DML API](http://apache.github.io/systemml/beginners-guide-caffe2dml.html): This API allows a model expressed in Caffe's proto format to be imported into SystemML. This API **doesnot** require Caffe to be installed on your SystemML.
3. Using the experimental [Keras2DML API](http://apache.github.io/systemml/beginners-guide-keras2dml.html): This API allows a model expressed in Keras's API to be imported into SystemML. However, this API requires Keras to be installed on your driver.

|                                                                                                      | NN library                                                                                                 | Caffe2DML                                                                                                     | Keras2DML                                                                       |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| External dependency                                                                                  | None                                                                                                       | None                                                                                                          | Keras                                                                           |
| Ability to add custom layers                                                                         | Yes                                                                                                        | No                                                                                                            | No                                                                              |
| The user needs to know                                                                               | [DML](http://apache.github.io/systemml/dml-language-reference)                                             | [Caffe's proto API](http://apache.github.io/systemml/reference-guide-caffe2dml#layers-supported-in-caffe2dml) | [Keras' API](https://keras.io/models/about-keras-models/)                       |
| Can be invoked using pyspark                                                                         | Yes. Please see [Python MLContext API](http://apache.github.io/systemml/spark-mlcontext-programming-guide) | Yes.                                                                                                          | Yes.                                                                            |
| Can be invoked using spark-shell                                                                     | Yes. Please see [Scala MLContext API](http://apache.github.io/systemml/spark-mlcontext-programming-guide)  | Limited support                                                                                               | No                                                                              |
| Can be invoked via command-line or JMLC API                                                          | Yes                                                                                                        | No                                                                                                            | No                                                                              |
| GPU and [native BLAS](http://apache.github.io/systemml/native-backend.html) support                  | Yes                                                                                                        | Yes                                                                                                           | Yes                                                                             |
| Part of SystemML's [mllearn](http://apache.github.io/systemml/python-reference.html#mllearn-api) API | No                                                                                                         | Yes                                                                                                           | Yes                                                                             |

## mllearn API

Before we go any further, let us briefly discuss the training and prediction functions in the mllearn API (i.e. Caffe2DML and Keras2DML).

### Training functions

<div class="codetabs">
<div data-lang="sklearn way" markdown="1">
{% highlight python %}
# Input: Two Python objects (X_train, y_train) of type numpy, pandas or scipy.
model.fit(X_train, y_train)
{% endhighlight %}
</div>
<div data-lang="mllib way" markdown="1">
{% highlight python %}
# Input: One LabeledPoint DataFrame with atleast two columns: features (of type Vector) and labels.
model.fit(X_df)
{% endhighlight %}
</div>
</div>

### Prediction functions

<div class="codetabs">
<div data-lang="sklearn way" markdown="1">
{% highlight python %}
# Input: One Python object (X_test) of type numpy, pandas or scipy.
model.predict(X_test)
# OR model.score(X_test, y_test)
{% endhighlight %}
</div>
<div data-lang="mllib way" markdown="1">
{% highlight python %}
# Input: One LabeledPoint DataFrame (df_test) with atleast one column: features (of type Vector).
model.transform(df_test)
{% endhighlight %}
</div>
</div>

Please note that when training using mllearn API (i.e. `model.fit(X_df)`), SystemML 
expects that labels have been converted to 1-based value.
This avoids unnecessary decoding overhead for large dataset if the label columns has already been decoded.
For scikit-learn API, there is no such requirement.

# Training Lenet on the MNIST dataset

Download the MNIST dataset using [mlxtend package](https://pypi.python.org/pypi/mlxtend).

```python
from mlxtend.data import mnist_data
import numpy as np
from sklearn.utils import shuffle
# Download the MNIST dataset
X, y = mnist_data()
X, y = shuffle(X, y)
# Split the data into training and test
n_samples = len(X)
X_train = X[:int(.9 * n_samples)]
y_train = y[:int(.9 * n_samples)]
X_test = X[int(.9 * n_samples):]
y_test = y[int(.9 * n_samples):]
```

<div class="codetabs">

<div data-lang="NN library" markdown="1">
{% highlight python %}
from systemml import MLContext, dml

ml = MLContext(sc)
ml.setStatistics(True)
# ml.setConfigProperty("sysml.native.blas", "auto")
# ml.setGPU(True).setForceGPU(True)
script = """
  source("nn/examples/mnist_lenet.dml") as mnist_lenet

  # Scale images to [-1,1], and one-hot encode the labels
  images = (images / 255) * 2 - 1
  n = nrow(images)
  labels = table(seq(1, n), labels+1, n, 10)

  # Split into training (4000 examples) and validation (4000 examples)
  X = images[501:nrow(images),]
  X_val = images[1:500,]
  y = labels[501:nrow(images),]
  y_val = labels[1:500,]

  # Train the model to produce weights & biases.
  [W1, b1, W2, b2, W3, b3, W4, b4] = mnist_lenet::train(X, y, X_val, y_val, C, Hin, Win, epochs)
"""
out = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4')
prog = (dml(script).input(images=X_train, labels=y_train.reshape((-1, 1)), epochs=1, C=1, Hin=28, Win=28)
                   .output(*out))

W1, b1, W2, b2, W3, b3, W4, b4 = ml.execute(prog).get(*out)

script_predict = """
  source("nn/examples/mnist_lenet.dml") as mnist_lenet

  # Scale images to [-1,1]
  X_test = (X_test / 255) * 2 - 1

  # Predict
  y_prob = mnist_lenet::predict(X_test, C, Hin, Win, W1, b1, W2, b2, W3, b3, W4, b4)
  y_pred = rowIndexMax(y_prob) - 1
"""
prog = (dml(script_predict).input(X_test=X_test, C=1, Hin=28, Win=28, W1=W1, b1=b1,
                                  W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)
                           .output("y_pred"))

y_pred = ml.execute(prog).get("y_pred").toNumPy()
{% endhighlight %}
</div>

<div data-lang="Caffe2DML" markdown="1">
{% highlight python %}
from systemml.mllearn import Caffe2DML
import urllib

# Download the Lenet network
urllib.urlretrieve('https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/mnist_lenet/lenet.proto', 'lenet.proto')
urllib.urlretrieve('https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/mnist_lenet/lenet_solver.proto', 'lenet_solver.proto')
# Train Lenet On MNIST using scikit-learn like API

# MNIST dataset contains 28 X 28 gray-scale (number of channel=1).
lenet = Caffe2DML(spark, solver='lenet_solver.proto', input_shape=(1, 28, 28))
lenet.setStatistics(True)
# lenet.setConfigProperty("sysml.native.blas", "auto")
# lenet.setGPU(True).setForceGPU(True)

# Since Caffe2DML is a mllearn API, it allows for scikit-learn like method for training.
lenet.fit(X_train, y_train)
# Either perform prediction: lenet.predict(X_test) or scoring:
lenet.score(X_test, y_test)
{% endhighlight %}
</div>

<div data-lang="Keras2DML" markdown="1">
{% highlight python %}
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout,Flatten
from keras import backend as K
from keras.models import Model
input_shape = (1,28,28) if K.image_data_format() == 'channels_first' else (28,28, 1)
keras_model = Sequential()
keras_model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, padding='same'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Flatten())
keras_model.add(Dense(512, activation='relu'))
keras_model.add(Dropout(0.5))
keras_model.add(Dense(10, activation='softmax'))
keras_model.summary()

# Scale the input features
scale = 0.00390625
X_train = X_train*scale
X_test = X_test*scale

from systemml.mllearn import Keras2DML
sysml_model = Keras2DML(spark, keras_model, input_shape=(1,28,28), weights='weights_dir')
# sysml_model.setConfigProperty("sysml.native.blas", "auto")
# sysml_model.setGPU(True).setForceGPU(True)
sysml_model.summary()
sysml_model.fit(X_train, y_train)
sysml_model.score(X_test, y_test)
{% endhighlight %}
</div>

</div>

# Prediction using a pretrained ResNet-50

<div class="codetabs">

<div data-lang="NN library" markdown="1">
{% highlight python %}
Will be added soon ...
{% endhighlight %}
</div>

<div data-lang="Caffe2DML" markdown="1">
{% highlight python %}
Will be added soon ...
{% endhighlight %}
</div>

<div data-lang="Keras2DML" markdown="1">
{% highlight python %}
from systemml.mllearn import Keras2DML
import systemml as sml
import keras, urllib
from PIL import Image
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50

keras_model = ResNet50(weights='imagenet',include_top=True,pooling='None',input_shape=(224,224,3))
keras_model.compile(optimizer='sgd', loss= 'categorical_crossentropy')

sysml_model = Keras2DML(spark,keras_model,input_shape=(3,224,224), weights='weights_dir', labels='https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/imagenet/labels.txt')
sysml_model.summary()
urllib.urlretrieve('https://upload.wikimedia.org/wikipedia/commons/f/f4/Cougar_sitting.jpg', 'test.jpg')
img_shape = (3, 224, 224)
input_image = sml.convertImageToNumPyArr(Image.open('test.jpg'), img_shape=img_shape)
sysml_model.predict(input_image)
{% endhighlight %}
</div>

</div>