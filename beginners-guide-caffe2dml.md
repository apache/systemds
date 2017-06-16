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

## Introduction

Caffe2DML is an **experimental API** that converts an Caffe specification to DML. 
It is designed to fit well into the mllearn framework and hence supports NumPy, Pandas as well as PySpark DataFrame.

## Examples

### Train Lenet on MNIST dataset

#### MNIST dataset

The MNIST dataset was constructed from two datasets of the US National Institute of Standards and Technology (NIST). The training set consists of handwritten digits from 250 different people, 50 percent high school students, and 50 percent employees from the Census Bureau. Note that the test set contains handwritten digits from different people following the same split.
In the below example, we are using mlxtend package to load the mnist dataset into Python NumPy arrays, but you are free to download it directly from http://yann.lecun.com/exdb/mnist/.

```bash
pip install mlxtend
```

#### Lenet network

Lenet is a simple convolutional neural network, proposed by Yann LeCun in 1998. It has 2 convolutions/pooling and fully connected layer. 
Similar to Caffe, the network has been modified to add dropout. 
For more detail, please see http://yann.lecun.com/exdb/lenet/

The [solver specification](https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/mnist_lenet/lenet_solver.proto)
specifies to Caffe2DML to use following configuration when generating the training DML script:  
- `type: "SGD", momentum: 0.9`: Stochastic Gradient Descent with momentum optimizer with `momentum=0.9`.
- `lr_policy: "exp", gamma: 0.95, base_lr: 0.01`: Use exponential decay learning rate policy (`base_lr * gamma ^ iter`).
- `display: 100`: Display training loss after every 100 iterations.
- `test_interval: 500`: Display validation loss after every 500 iterations.
- `test_iter: 10`: Validation data size = 10 * BATCH_SIZE.
 

```python
from mlxtend.data import mnist_data
import numpy as np
from sklearn.utils import shuffle
import urllib
from systemml.mllearn import Caffe2DML

# Download the MNIST dataset
X, y = mnist_data()
X, y = shuffle(X, y)

# Split the data into training and test
n_samples = len(X)
X_train = X[:int(.9 * n_samples)]
y_train = y[:int(.9 * n_samples)]
X_test = X[int(.9 * n_samples):]
y_test = y[int(.9 * n_samples):]

# Download the Lenet network
urllib.urlretrieve('https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/mnist_lenet/lenet.proto', 'lenet.proto')
urllib.urlretrieve('https://raw.githubusercontent.com/apache/systemml/master/scripts/nn/examples/caffe2dml/models/mnist_lenet/lenet_solver.proto', 'lenet_solver.proto')

# Train Lenet On MNIST using scikit-learn like API
# MNIST dataset contains 28 X 28 gray-scale (number of channel=1).
lenet = Caffe2DML(sqlCtx, solver='lenet_solver.proto', input_shape=(1, 28, 28))

# debug=True prints will print the generated DML script along with classification report. Please donot test this flag in production.
lenet.set(debug=True)

# If you want to see the statistics as well as the plan
lenet.setStatistics(True).setExplain(True)

# If you want to force GPU execution. Please make sure the required dependency are available.  
# lenet.setGPU(True).setForceGPU(True)
# Example usage of train_algo, test_algo. Assume 2 gpus on driver
# lenet.set(train_algo="allreduce_parallel_batches", test_algo="minibatch", parallel_batches=2)

# (Optional but recommended) Enable native BLAS. 
lenet.setConfigProperty("native.blas", "auto")

# In case you want to enable experimental feature such as codegen
# lenet.setConfigProperty("codegen.enabled", "true").setConfigProperty("codegen.plancache", "true")

# Since Caffe2DML is a mllearn API, it allows for scikit-learn like method for training.
lenet.fit(X_train, y_train)
lenet.predict(X_test)
```

For more detail on enabling native BLAS, please see the documentation for the [native backend](http://apache.github.io/systemml/native-backend).

Common settings for `train_algo` and `test_algo` parameters:

|                                                                          | PySpark script                                                                                                                           | Changes to Network/Solver                                              |
|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| Single-node CPU execution (similar to Caffe with solver_mode: CPU)       | `caffe2dml.set(train_algo="minibatch", test_algo="minibatch")`                                                                           | Ensure that `batch_size` is set to appropriate value (for example: 64) |
| Single-node single-GPU execution                                         | `caffe2dml.set(train_algo="minibatch", test_algo="minibatch").setGPU(True).setForceGPU(True)`                                            | Ensure that `batch_size` is set to appropriate value (for example: 64) |
| Single-node multi-GPU execution (similar to Caffe with solver_mode: GPU) | `caffe2dml.set(train_algo="allreduce_parallel_batches", test_algo="minibatch", parallel_batches=num_gpu).setGPU(True).setForceGPU(True)` | Ensure that `batch_size` is set to appropriate value (for example: 64) |
| Distributed prediction                                                   | `caffe2dml.set(test_algo="allreduce")`                                                                                                   |                                                                        |
| Distributed synchronous training                                         | `caffe2dml.set(train_algo="allreduce_parallel_batches", parallel_batches=num_cluster_cores)`                                             | Ensure that `batch_size` is set to appropriate value (for example: 64) |

## Frequently asked questions

#### What is the purpose of Caffe2DML API ?

Most deep learning experts are more likely to be familiar with the Caffe's specification
rather than DML language. For these users, the Caffe2DML API reduces the learning curve to using SystemML.
Instead of requiring the users to write a DML script for training, fine-tuning and testing the model,
Caffe2DML takes as an input a network and solver specified in the Caffe specification
and automatically generates the corresponding DML.

#### With Caffe2DML, does SystemML now require Caffe to be installed ?

Absolutely not. We only support Caffe's API for convenience of the user as stated above.
Since the Caffe's API is specified in the protobuf format, we are able to generate the java parser files
and donot require Caffe to be installed. This is also true for Tensorboard feature of Caffe2DML. 

```
Dml.g4      ---> antlr  ---> DmlLexer.java, DmlListener.java, DmlParser.java ---> parse foo.dml
caffe.proto ---> protoc ---> target/generated-sources/caffe/Caffe.java       ---> parse caffe_network.proto, caffe_solver.proto 
```

Again, the SystemML engine doesnot invoke (or depend on) Caffe and TensorFlow for any of its runtime operators.
Since the grammar files for the respective APIs (i.e. `caffe.proto`) are used by SystemML, 
we include their licenses in our jar files.

#### How can I speedup the training with Caffe2DML ?

- Enable native BLAS to improve the performance of CP convolution and matrix multiplication operators.
If you are using OpenBLAS, please ensure that it was built with `USE_OPENMP` flag turned on.
For more detail see http://apache.github.io/systemml/native-backend

```python
caffe2dmlObject.setConfigProperty("native.blas", "auto")
```

- Turn on the experimental codegen feature. This should help reduce unnecessary allocation cost after every binary operation.

```python
caffe2dmlObject.setConfigProperty("codegen.enabled", "true").setConfigProperty("codegen.plancache", "true")
```

- Tuned the [Garbage Collector](http://spark.apache.org/docs/latest/tuning.html#garbage-collection-tuning). 

- Enable GPU support (described below).

#### How to enable GPU support in Caffe2DML ?

To be consistent with other mllearn algorithms, we recommend that you use following method instead of setting 
the `solver_mode` in solver file.

```python
# The below method tells SystemML optimizer to use a GPU-enabled instruction if the operands fit in the GPU memory 
caffe2dmlObject.setGPU(True)
# The below method tells SystemML optimizer to always use a GPU-enabled instruction irrespective of the memory requirement
caffe2dmlObject.setForceGPU(True)
```

#### What is lr_policy in the solver specification ?

The parameter `lr_policy` specifies the learning rate decay policy. Caffe2DML supports following policies:
- `fixed`: always return `base_lr`.
- `step`: return `base_lr * gamma ^ (floor(iter / step))`
- `exp`: return `base_lr * gamma ^ iter`
- `inv`: return `base_lr * (1 + gamma * iter) ^ (- power)`
- `poly`: the effective learning rate follows a polynomial decay, to be zero by the max_iter. return `base_lr (1 - iter/max_iter) ^ (power)`
- `sigmoid`: the effective learning rate follows a sigmod decay return b`ase_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))`
      
#### How to set batch size ?

Batch size is set in `data_param` of the Data layer:

```
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  data_param {
    source: "mnist_train"
    batch_size: 64
    backend: LMDB
  }
}
```
	
#### How to set maximum number of iterations for training ?

The maximum number of iterations can be set in the solver specification

```bash
# The maximum number of iterations
max_iter: 2000
```

#### How to set the size of the validation dataset ?

The size of the validation dataset is determined by the parameters `test_iter` and the batch size. For example: If the batch size is 64 and 
`test_iter` is 10, then the validation size is 640. This setting generates following DML code internally:

```python
num_images = nrow(y_full)
BATCH_SIZE = 64
num_validation = 10 * BATCH_SIZE
X = X_full[(num_validation+1):num_images,]; y = y_full[(num_validation+1):num_images,]
X_val = X_full[1:num_validation,]; y_val = y_full[1:num_validation,]
num_images = nrow(y)
``` 

#### How to monitor loss via command-line ?

To monitor loss, please set following parameters in the solver specification

```
# Display training loss and accuracy every 100 iterations
display: 100
# Carry out validation every 500 training iterations and display validation loss and accuracy.
test_iter: 10
test_interval: 500
```

#### How to pass a single jpeg image to Caffe2DML for prediction ?

To convert a jpeg into NumPy matrix, you can use the [pillow package](https://pillow.readthedocs.io/) and 
SystemML's  `convertImageToNumPyArr` utility function. The below pyspark code demonstrates the usage:
 
```python
from PIL import Image
import systemml as sml
from systemml.mllearn import Caffe2DML
img_shape = (3, 224, 224)
input_image = sml.convertImageToNumPyArr(Image.open(img_file_path), img_shape=img_shape)
resnet = Caffe2DML(sqlCtx, solver='ResNet_50_solver.proto', weights='ResNet_50_pretrained_weights', input_shape=img_shape)
resnet.predict(input_image)
```

#### How to prepare a directory of jpeg images for training with Caffe2DML ?

The below pyspark code assumes that the input dataset has 2 labels `cat` and `dogs` and the filename has these labels as prefix.
We iterate through the directory and convert each jpeg image into pyspark.ml.linalg.Vector using pyspark.
These vectors are stored as DataFrame and randomized using Spark SQL's `orderBy(rand())` function.
The DataFrame is then saved in parquet format to reduce the cost of preprocessing for repeated training.

```python
from systemml.mllearn import Caffe2DML
from pyspark.sql import SQLContext
import numpy as np
import urllib, os, scipy.ndimage
from pyspark.ml.linalg import Vectors
from pyspark import StorageLevel
import systemml as sml
from pyspark.sql.functions import rand 
# ImageNet specific parameters
img_shape = (3, 224, 224)
train_dir = '/home/biuser/dogs_vs_cats/train'
def getLabelFeatures(filename):
	from PIL import Image
	vec = Vectors.dense(sml.convertImageToNumPyArr(Image.open(os.path.join(train_dir, filename)), img_shape=img_shape)[0,:])
	if filename.lower().startswith('cat'):
		return (1, vec)
	elif filename.lower().startswith('dog'):
		return (2, vec)
	else:
		raise ValueError('Expected the filename to start with either cat or dog')
list_jpeg_files = os.listdir(train_dir)
# 10 files per partition
train_df = sc.parallelize(list_jpeg_files, int(len(list_jpeg_files)/10)).map(lambda filename : getLabelFeatures(filename)).toDF(['label', 'features']).orderBy(rand())
# Optional: but helps seperates conversion-related from training
# Alternatively, this dataframe can be passed directly to `caffe2dml_model.fit(train_df)`
train_df.write.parquet('kaggle-cats-dogs.parquet')
```

#### Can I use Caffe2DML via Scala ?

Though we recommend using Caffe2DML via its Python interfaces, it is possible to use it by creating an object of the class
`org.apache.sysml.api.dl.Caffe2DML`. It is important to note that Caffe2DML's scala API is packaged in `systemml-*-extra.jar`.


#### How can I view the script generated by Caffe2DML ?

To view the generated DML script (and additional debugging information), please set the `debug` parameter to True.

```python
caffe2dmlObject.set(debug=True)
```
