---
layout: global
title: Beginner's Guide for Caffe2DML users
description: Beginner's Guide for Caffe2DML users
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

Caffe2DML is an **experimental API** that converts a Caffe specification to DML. 
It is designed to fit well into the mllearn framework and hence supports NumPy, Pandas as well as PySpark DataFrame.

# Training Lenet 

To create a Caffe2DML object, one needs to create a solver and network file that conforms 
to the [Caffe specification](http://caffe.berkeleyvision.org/).
In this example, we will train Lenet which is a simple convolutional neural network, proposed by Yann LeCun in 1998. 
It has 2 convolutions/pooling and fully connected layer. 
Similar to Caffe, the network has been modified to add dropout. 
For more detail, please see [http://yann.lecun.com/exdb/lenet/](http://yann.lecun.com/exdb/lenet/).

The [solver specification](https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/mnist_lenet/lenet_solver.proto)
specifies to Caffe2DML to use following configuration when generating the training DML script:  
- `type: "SGD", momentum: 0.9`: Stochastic Gradient Descent with momentum optimizer with `momentum=0.9`.
- `lr_policy: "exp", gamma: 0.95, base_lr: 0.01`: Use exponential decay learning rate policy (`base_lr * gamma ^ iter`).
- `display: 100`: Display training loss after every 100 iterations.
- `test_interval: 500`: Display validation loss after every 500 iterations.
- `test_iter: 10`: Validation data size = 10 * BATCH_SIZE.

```python
from systemml.mllearn import Caffe2DML
import urllib

# Download the Lenet network
urllib.urlretrieve('https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/mnist_lenet/lenet.proto', 'lenet.proto')
urllib.urlretrieve('https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/mnist_lenet/lenet_solver.proto', 'lenet_solver.proto')
# Train Lenet On MNIST using scikit-learn like API

# MNIST dataset contains 28 X 28 gray-scale (number of channel=1).
lenet = Caffe2DML(spark, solver='lenet_solver.proto', input_shape=(1, 28, 28))
lenet.summary()
```

Output:

```
+-----+---------------+--------------+------------+---------+-----------+---------+
| Name|           Type|        Output|      Weight|     Bias|        Top|   Bottom|
+-----+---------------+--------------+------------+---------+-----------+---------+
|mnist|           Data| (, 1, 28, 28)|            |         |mnist,mnist|         |
|conv1|    Convolution|(, 32, 28, 28)|   [32 X 25]| [32 X 1]|      conv1|    mnist|
|relu1|           ReLU|(, 32, 28, 28)|            |         |      relu1|    conv1|
|pool1|        Pooling|(, 32, 14, 14)|            |         |      pool1|    relu1|
|conv2|    Convolution|(, 64, 14, 14)|  [64 X 800]| [64 X 1]|      conv2|    pool1|
|relu2|           ReLU|(, 64, 14, 14)|            |         |      relu2|    conv2|
|pool2|        Pooling|  (, 64, 7, 7)|            |         |      pool2|    relu2|
|  ip1|   InnerProduct| (, 512, 1, 1)|[3136 X 512]|[1 X 512]|        ip1|    pool2|
|relu3|           ReLU| (, 512, 1, 1)|            |         |      relu3|      ip1|
|drop1|        Dropout| (, 512, 1, 1)|            |         |      drop1|    relu3|
|  ip2|   InnerProduct|  (, 10, 1, 1)|  [512 X 10]| [1 X 10]|        ip2|    drop1|
| loss|SoftmaxWithLoss|  (, 10, 1, 1)|            |         |       loss|ip2,mnist|
+-----+---------------+--------------+------------+---------+-----------+---------+
``` 

To train the above lenet model, we use the MNIST dataset. 
The MNIST dataset was constructed from two datasets of the US National Institute of Standards and Technology (NIST). 
The training set consists of handwritten digits from 250 different people, 50 percent high school students, and 50 percent employees from the Census Bureau. Note that the test set contains handwritten digits from different people following the same split.
In this example, we are using mlxtend package to load the mnist dataset into Python NumPy arrays, but you are free to download it directly from http://yann.lecun.com/exdb/mnist/.

```bash
pip install mlxtend
```

We first split the MNIST dataset into train and test.  

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

Finally, we use the training and test dataset to perform training and prediction using scikit-learn like API.

```python
# Since Caffe2DML is a mllearn API, it allows for scikit-learn like method for training.
lenet.fit(X_train, y_train)
# Either perform prediction: lenet.predict(X_test) or scoring:
lenet.score(X_test, y_test)
```

Output:
```
Iter:100, training loss:0.189008481420049, training accuracy:92.1875
Iter:200, training loss:0.21657020576713149, training accuracy:96.875
Iter:300, training loss:0.05780939180052287, training accuracy:98.4375
Iter:400, training loss:0.03406193840071965, training accuracy:100.0
Iter:500, training loss:0.02847187709112875, training accuracy:100.0
Iter:500, validation loss:222.736109642486, validation accuracy:96.49077868852459
Iter:600, training loss:0.04867848427394318, training accuracy:96.875
Iter:700, training loss:0.043060905384304224, training accuracy:98.4375
Iter:800, training loss:0.01861298388336358, training accuracy:100.0
Iter:900, training loss:0.03495462005933769, training accuracy:100.0
Iter:1000, training loss:0.04598737325942163, training accuracy:98.4375
Iter:1000, validation loss:180.04232316810746, validation accuracy:97.28483606557377
Iter:1100, training loss:0.05630274512793694, training accuracy:98.4375
Iter:1200, training loss:0.027278141291535066, training accuracy:98.4375
Iter:1300, training loss:0.04356275106270366, training accuracy:98.4375
Iter:1400, training loss:0.00780793048139091, training accuracy:100.0
Iter:1500, training loss:0.004135965492374173, training accuracy:100.0
Iter:1500, validation loss:156.61636761709374, validation accuracy:97.48975409836065
Iter:1600, training loss:0.007939063305475983, training accuracy:100.0
Iter:1700, training loss:0.0025769653351162196, training accuracy:100.0
Iter:1800, training loss:0.0023251742357435204, training accuracy:100.0
Iter:1900, training loss:0.0016795711023936644, training accuracy:100.0
Iter:2000, training loss:0.03676045262879483, training accuracy:98.4375
Iter:2000, validation loss:173.66147359346, validation accuracy:97.48975409836065
0.97399999999999998
```

# Additional Configuration

- Print the generated DML script along with classification report:  `lenet.set(debug=True)`
- Print the heavy hitters instruction and the execution plan (advanced users): `lenet.setStatistics(True).setExplain(True)`
- (Optional but recommended) Enable [native BLAS](http://apache.github.io/systemml/native-backend): `lenet.setConfigProperty("sysml.native.blas", "auto")`
- Enable experimental feature such as codegen: `lenet.setConfigProperty("sysml.codegen.enabled", "true").setConfigProperty("sysml.codegen.plancache", "true")`
- Force GPU execution (please make sure the required jcuda dependency are included): lenet.setGPU(True).setForceGPU(True)

Unlike Caffe where default train and test algorithm is `minibatch`, you can specify the
algorithm using the parameters `train_algo` and `test_algo` (valid values are: `minibatch`, `allreduce_parallel_batches`, 
and `allreduce`). Here are some common settings:

|                                                                          | PySpark script                                                                                                                           | Changes to Network/Solver                                              |
|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| Single-node CPU execution (similar to Caffe with solver_mode: CPU)       | `lenet.set(train_algo="minibatch", test_algo="minibatch")`                                                                               | Ensure that `batch_size` is set to appropriate value (for example: 64) |
| Single-node single-GPU execution                                         | `lenet.set(train_algo="minibatch", test_algo="minibatch").setGPU(True).setForceGPU(True)`                                                | Ensure that `batch_size` is set to appropriate value (for example: 64) |
| Single-node multi-GPU execution (similar to Caffe with solver_mode: GPU) | `lenet.set(train_algo="allreduce_parallel_batches", test_algo="minibatch", parallel_batches=num_gpu).setGPU(True).setForceGPU(True)`     | Ensure that `batch_size` is set to appropriate value (for example: 64) |
| Distributed prediction                                                   | `lenet.set(test_algo="allreduce")`                                                                                                       |                                                                        |
| Distributed synchronous training                                         | `lenet.set(train_algo="allreduce_parallel_batches", parallel_batches=num_cluster_cores)`                                                 | Ensure that `batch_size` is set to appropriate value (for example: 64) |

# Saving the trained model

```python
lenet.fit(X_train, y_train)
lenet.save('trained_weights')
new_lenet = Caffe2DML(spark, solver='lenet_solver.proto', input_shape=(1, 28, 28))
new_lenet.load('trained_weights')
new_lenet.score(X_test, y_test)
```

# Loading a pretrained caffemodel

We provide a converter utility to convert `.caffemodel` trained using Caffe to SystemML format.

```python
# First download deploy file and caffemodel
import urllib
urllib.urlretrieve('https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/imagenet/vgg19/VGG_ILSVRC_19_layers_deploy.proto', 'VGG_ILSVRC_19_layers_deploy.proto')
urllib.urlretrieve('http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel', 'VGG_ILSVRC_19_layers.caffemodel')
# Save the weights into trained_vgg_weights directory
import systemml as sml
sml.convert_caffemodel(sc, 'VGG_ILSVRC_19_layers_deploy.proto', 'VGG_ILSVRC_19_layers.caffemodel',  'trained_vgg_weights')
```

We can then use the `trained_vgg_weights` directory for performing prediction or fine-tuning.

```python
# Download the VGG network
urllib.urlretrieve('https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/imagenet/vgg19/VGG_ILSVRC_19_layers_network.proto', 'VGG_ILSVRC_19_layers_network.proto')
urllib.urlretrieve('https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/imagenet/vgg19/VGG_ILSVRC_19_layers_solver.proto', 'VGG_ILSVRC_19_layers_solver.proto')
# Storing the labels.txt in the weights directory allows predict to return a label (for example: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor') rather than the column index of one-hot encoded vector (for example: 287).
urllib.urlretrieve('https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/imagenet/labels.txt', os.path.join('trained_vgg_weights', 'labels.txt'))
from systemml.mllearn import Caffe2DML
vgg = Caffe2DML(sqlCtx, solver='VGG_ILSVRC_19_layers_solver.proto', input_shape=(3, 224, 224))
vgg.load('trained_vgg_weights')
# We can then perform prediction:
from PIL import Image
X_test = sml.convertImageToNumPyArr(Image.open('test.jpg'), img_shape=(3, 224, 224))
vgg.predict(X_test)
# OR Fine-Tuning: vgg.fit(X_train, y_train)
```

Please see [Caffe2DML's reference guide](http://apache.github.io/systemml/reference-guide-caffe2dml) for more details.