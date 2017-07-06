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

### Training Lenet 

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

### Additional Configuration

- Print the generated DML script along with classification report:  `lenet.set(debug=True)`
- Print the heavy hitters instruction and the execution plan (advanced users): `lenet.setStatistics(True).setExplain(True)`
- (Optional but recommended) Enable [native BLAS](http://apache.github.io/systemml/native-backend): `lenet.setConfigProperty("native.blas", "auto")`
- Enable experimental feature such as codegen: `lenet.setConfigProperty("codegen.enabled", "true").setConfigProperty("codegen.plancache", "true")`
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

### Saving the trained model

```python
lenet.fit(X_train, y_train)
lenet.save('trained_weights')
new_lenet = Caffe2DML(spark, solver='lenet_solver.proto', input_shape=(1, 28, 28))
new_lenet.load('trained_weights')
new_lenet.score(X_test, y_test)
```

### Loading a pretrained caffemodel

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

An alternative way to load images into a PySpark DataFrame for prediction, is to use MLLib's LabeledPoint class:

```python
list_jpeg_files = os.listdir(train_dir)
train_df = sc.parallelize(list_jpeg_files, int(len(list_jpeg_files)/10)).map(lambda filename : LabeledPoint(0, sml.convertImageToNumPyArr(Image.open(os.path.join(train_dir, filename)), img_shape=img_shape)[0,:])).toDF().select('features')
# Note: convertVectorColumnsToML has an additional serialization cost
train_df = MLUtils.convertVectorColumnsToML(train_df)
```
 

#### Can I use Caffe2DML via Scala ?

Though we recommend using Caffe2DML via its Python interfaces, it is possible to use it by creating an object of the class
`org.apache.sysml.api.dl.Caffe2DML`. It is important to note that Caffe2DML's scala API is packaged in `systemml-*-extra.jar`.

#### How can I get summary information of my network ?
 

```python
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

#### How can I view the script generated by Caffe2DML ?

To view the generated DML script (and additional debugging information), please set the `debug` parameter to True.

```python
lenet.set(debug=True)
```

Output:
```
001|debug = TRUE
002|source("nn/layers/softmax.dml") as softmax
003|source("nn/layers/cross_entropy_loss.dml") as cross_entropy_loss
004|source("nn/layers/conv2d_builtin.dml") as conv2d_builtin
005|source("nn/layers/relu.dml") as relu
006|source("nn/layers/max_pool2d_builtin.dml") as max_pool2d_builtin
007|source("nn/layers/affine.dml") as affine
008|source("nn/layers/dropout.dml") as dropout
009|source("nn/optim/sgd_momentum.dml") as sgd_momentum
010|source("nn/layers/l2_reg.dml") as l2_reg
011|X_full_path = ifdef($X, " ")
012|X_full = read(X_full_path)
013|y_full_path = ifdef($y, " ")
014|y_full = read(y_full_path)
015|num_images = nrow(y_full)
016|# Convert to one-hot encoding (Assumption: 1-based labels)
017|y_full = table(seq(1,num_images,1), y_full, num_images, 10)
018|weights = ifdef($weights, " ")
019|# Initialize the layers and solvers
020|X_full = X_full * 0.00390625
021|BATCH_SIZE = 64
022|[conv1_weight,conv1_bias] = conv2d_builtin::init(32,1,5,5)
023|[conv2_weight,conv2_bias] = conv2d_builtin::init(64,32,5,5)
024|[ip1_weight,ip1_bias] = affine::init(3136,512)
025|[ip2_weight,ip2_bias] = affine::init(512,10)
026|conv1_weight_v = sgd_momentum::init(conv1_weight)
027|conv1_bias_v = sgd_momentum::init(conv1_bias)
028|conv2_weight_v = sgd_momentum::init(conv2_weight)
029|conv2_bias_v = sgd_momentum::init(conv2_bias)
030|ip1_weight_v = sgd_momentum::init(ip1_weight)
031|ip1_bias_v = sgd_momentum::init(ip1_bias)
032|ip2_weight_v = sgd_momentum::init(ip2_weight)
033|ip2_bias_v = sgd_momentum::init(ip2_bias)
034|num_validation = 10 * BATCH_SIZE
035|# Sanity check to ensure that validation set is not too large
036|if(num_validation > ceil(0.3 * num_images)) {
037|    max_test_iter = floor(ceil(0.3 * num_images) / BATCH_SIZE)
038|    stop("Too large validation size. Please reduce test_iter to " + max_test_iter)
039|}
040|X = X_full[(num_validation+1):num_images,]; y = y_full[(num_validation+1):num_images,]; X_val = X_full[1:num_validation,]; y_val = y_full[1:num_validation,]; num_images = nrow(y)
041|num_iters_per_epoch = ceil(num_images / BATCH_SIZE)
042|max_epochs = ceil(2000 / num_iters_per_epoch)
043|iter = 0
044|lr = 0.01
045|for(e in 1:max_epochs) {
046|    for(i in 1:num_iters_per_epoch) {
047|            beg = ((i-1) * BATCH_SIZE) %% num_images + 1; end = min(beg + BATCH_SIZE - 1, num_images); Xb = X[beg:end,]; yb = y[beg:end,];
048|            iter = iter + 1
049|            # Perform forward pass
050|            [out3,ignoreHout_3,ignoreWout_3] = conv2d_builtin::forward(Xb,conv1_weight,conv1_bias,1,28,28,5,5,1,1,2,2)
051|            out4 = relu::forward(out3)
052|            [out5,ignoreHout_5,ignoreWout_5] = max_pool2d_builtin::forward(out4,32,28,28,2,2,2,2,0,0)
053|            [out6,ignoreHout_6,ignoreWout_6] = conv2d_builtin::forward(out5,conv2_weight,conv2_bias,32,14,14,5,5,1,1,2,2)
054|            out7 = relu::forward(out6)
055|            [out8,ignoreHout_8,ignoreWout_8] = max_pool2d_builtin::forward(out7,64,14,14,2,2,2,2,0,0)
056|            out9 = affine::forward(out8,ip1_weight,ip1_bias)
057|            out10 = relu::forward(out9)
058|            [out11,mask11] = dropout::forward(out10,0.5,-1)
059|            out12 = affine::forward(out11,ip2_weight,ip2_bias)
060|            out13 = softmax::forward(out12)
061|            # Perform backward pass
062|            dProbs = cross_entropy_loss::backward(out13,yb); dOut13 = softmax::backward(dProbs,out12); dOut13_12 = dOut13; dOut13_2 = dOut13;
063|            [dOut12,ip2_dWeight,ip2_dBias] = affine::backward(dOut13_12,out11,ip2_weight,ip2_bias); dOut12_11 = dOut12;
064|            dOut11 = dropout::backward(dOut12_11,out10,0.5,mask11); dOut11_10 = dOut11;
065|            dOut10 = relu::backward(dOut11_10,out9); dOut10_9 = dOut10;
066|            [dOut9,ip1_dWeight,ip1_dBias] = affine::backward(dOut10_9,out8,ip1_weight,ip1_bias); dOut9_8 = dOut9;
067|            dOut8 = max_pool2d_builtin::backward(dOut9_8,7,7,out7,64,14,14,2,2,2,2,0,0); dOut8_7 = dOut8;
068|            dOut7 = relu::backward(dOut8_7,out6); dOut7_6 = dOut7;
069|            [dOut6,conv2_dWeight,conv2_dBias] = conv2d_builtin::backward(dOut7_6,14,14,out5,conv2_weight,conv2_bias,32,14,14,5,5,1,1,2,2); dOut6_5 = dOut6;
070|            dOut5 = max_pool2d_builtin::backward(dOut6_5,14,14,out4,32,28,28,2,2,2,2,0,0); dOut5_4 = dOut5;
071|            dOut4 = relu::backward(dOut5_4,out3); dOut4_3 = dOut4;
072|            [dOut3,conv1_dWeight,conv1_dBias] = conv2d_builtin::backward(dOut4_3,28,28,Xb,conv1_weight,conv1_bias,1,28,28,5,5,1,1,2,2); dOut3_2 = dOut3;
073|            # Update the parameters
074|            conv1_dWeight_reg = l2_reg::backward(conv1_weight, 5.000000237487257E-4)
075|            conv1_dWeight = conv1_dWeight + conv1_dWeight_reg
076|            [conv1_weight,conv1_weight_v] = sgd_momentum::update(conv1_weight,conv1_dWeight,(lr * 1.0),0.8999999761581421,conv1_weight_v)
077|            [conv1_bias,conv1_bias_v] = sgd_momentum::update(conv1_bias,conv1_dBias,(lr * 2.0),0.8999999761581421,conv1_bias_v)
078|            conv2_dWeight_reg = l2_reg::backward(conv2_weight, 5.000000237487257E-4)
079|            conv2_dWeight = conv2_dWeight + conv2_dWeight_reg
080|            [conv2_weight,conv2_weight_v] = sgd_momentum::update(conv2_weight,conv2_dWeight,(lr * 1.0),0.8999999761581421,conv2_weight_v)
081|            [conv2_bias,conv2_bias_v] = sgd_momentum::update(conv2_bias,conv2_dBias,(lr * 2.0),0.8999999761581421,conv2_bias_v)
082|            ip1_dWeight_reg = l2_reg::backward(ip1_weight, 5.000000237487257E-4)
083|            ip1_dWeight = ip1_dWeight + ip1_dWeight_reg
084|            [ip1_weight,ip1_weight_v] = sgd_momentum::update(ip1_weight,ip1_dWeight,(lr * 1.0),0.8999999761581421,ip1_weight_v)
085|            [ip1_bias,ip1_bias_v] = sgd_momentum::update(ip1_bias,ip1_dBias,(lr * 2.0),0.8999999761581421,ip1_bias_v)
086|            ip2_dWeight_reg = l2_reg::backward(ip2_weight, 5.000000237487257E-4)
087|            ip2_dWeight = ip2_dWeight + ip2_dWeight_reg
088|            [ip2_weight,ip2_weight_v] = sgd_momentum::update(ip2_weight,ip2_dWeight,(lr * 1.0),0.8999999761581421,ip2_weight_v)
089|            [ip2_bias,ip2_bias_v] = sgd_momentum::update(ip2_bias,ip2_dBias,(lr * 2.0),0.8999999761581421,ip2_bias_v)
090|            # Compute training loss & accuracy
091|            if(iter  %% 100 == 0) {
092|                    loss = 0
093|                    accuracy = 0
094|                    tmp_loss = cross_entropy_loss::forward(out13,yb)
095|                    loss = loss + tmp_loss
096|                    true_yb = rowIndexMax(yb)
097|                    predicted_yb = rowIndexMax(out13)
098|                    accuracy = mean(predicted_yb == true_yb)*100
099|                    training_loss = loss
100|                    training_accuracy = accuracy
101|                    print("Iter:" + iter + ", training loss:" + training_loss + ", training accuracy:" + training_accuracy)
102|                    if(debug) {
103|                            num_rows_error_measures = min(10, ncol(yb))
104|                            error_measures = matrix(0, rows=num_rows_error_measures, cols=5)
105|                            for(class_i in 1:num_rows_error_measures) {
106|                                    tp = sum( (true_yb == predicted_yb) * (true_yb == class_i) )
107|                                    tp_plus_fp = sum( (predicted_yb == class_i) )
108|                                    tp_plus_fn = sum( (true_yb == class_i) )
109|                                    precision = tp / tp_plus_fp
110|                                    recall = tp / tp_plus_fn
111|                                    f1Score = 2*precision*recall / (precision+recall)
112|                                    error_measures[class_i,1] = class_i
113|                                    error_measures[class_i,2] = precision
114|                                    error_measures[class_i,3] = recall
115|                                    error_measures[class_i,4] = f1Score
116|                                    error_measures[class_i,5] = tp_plus_fn
117|                            }
118|                            print("class    \tprecision\trecall  \tf1-score\tnum_true_labels\n" + toString(error_measures, decimal=7, sep="\t"))
119|                    }
120|            }
121|            # Compute validation loss & accuracy
122|            if(iter  %% 500 == 0) {
123|                    loss = 0
124|                    accuracy = 0
125|                    validation_loss = 0
126|                    validation_accuracy = 0
127|                    for(iVal in 1:num_iters_per_epoch) {
128|                            beg = ((iVal-1) * BATCH_SIZE) %% num_validation + 1; end = min(beg + BATCH_SIZE - 1, num_validation); Xb = X_val[beg:end,]; yb = y_val[beg:end,];
129|                            # Perform forward pass
130|                            [out3,ignoreHout_3,ignoreWout_3] = conv2d_builtin::forward(Xb,conv1_weight,conv1_bias,1,28,28,5,5,1,1,2,2)
131|                            out4 = relu::forward(out3)
132|                            [out5,ignoreHout_5,ignoreWout_5] = max_pool2d_builtin::forward(out4,32,28,28,2,2,2,2,0,0)
133|                            [out6,ignoreHout_6,ignoreWout_6] = conv2d_builtin::forward(out5,conv2_weight,conv2_bias,32,14,14,5,5,1,1,2,2)
134|                            out7 = relu::forward(out6)
135|                            [out8,ignoreHout_8,ignoreWout_8] = max_pool2d_builtin::forward(out7,64,14,14,2,2,2,2,0,0)
136|                            out9 = affine::forward(out8,ip1_weight,ip1_bias)
137|                            out10 = relu::forward(out9)
138|                            [out11,mask11] = dropout::forward(out10,0.5,-1)
139|                            out12 = affine::forward(out11,ip2_weight,ip2_bias)
140|                            out13 = softmax::forward(out12)
141|                            tmp_loss = cross_entropy_loss::forward(out13,yb)
142|                            loss = loss + tmp_loss
143|                            true_yb = rowIndexMax(yb)
144|                            predicted_yb = rowIndexMax(out13)
145|                            accuracy = mean(predicted_yb == true_yb)*100
146|                            validation_loss = validation_loss + loss
147|                            validation_accuracy = validation_accuracy + accuracy
148|                    }
149|                    validation_accuracy = validation_accuracy / num_iters_per_epoch
150|                    print("Iter:" + iter + ", validation loss:" + validation_loss + ", validation accuracy:" + validation_accuracy)
151|            }
152|    }
153|    # Learning rate
154|    lr = (0.009999999776482582 * 0.949999988079071^e)
155|}

Iter:100, training loss:0.24014199350958168, training accuracy:87.5
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       3.0000000
2.0000000       1.0000000       1.0000000       1.0000000       8.0000000
3.0000000       0.8888889       0.8888889       0.8888889       9.0000000
4.0000000       0.7500000       0.7500000       0.7500000       4.0000000
5.0000000       0.7500000       1.0000000       0.8571429       3.0000000
6.0000000       0.8333333       1.0000000       0.9090909       5.0000000
7.0000000       1.0000000       1.0000000       1.0000000       8.0000000
8.0000000       0.8571429       0.7500000       0.8000000       8.0000000
9.0000000       1.0000000       0.5714286       0.7272727       7.0000000
10.0000000      0.7272727       0.8888889       0.8000000       9.0000000

Iter:200, training loss:0.09555593867171894, training accuracy:98.4375
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       10.0000000
2.0000000       1.0000000       1.0000000       1.0000000       3.0000000
3.0000000       1.0000000       1.0000000       1.0000000       9.0000000
4.0000000       1.0000000       1.0000000       1.0000000       6.0000000
5.0000000       1.0000000       1.0000000       1.0000000       7.0000000
6.0000000       1.0000000       1.0000000       1.0000000       8.0000000
7.0000000       1.0000000       0.6666667       0.8000000       3.0000000
8.0000000       1.0000000       1.0000000       1.0000000       9.0000000
9.0000000       0.8571429       1.0000000       0.9230769       6.0000000
10.0000000      1.0000000       1.0000000       1.0000000       3.0000000

Iter:300, training loss:0.058686794512570216, training accuracy:98.4375
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       6.0000000
2.0000000       1.0000000       1.0000000       1.0000000       9.0000000
3.0000000       1.0000000       1.0000000       1.0000000       4.0000000
4.0000000       1.0000000       1.0000000       1.0000000       8.0000000
5.0000000       1.0000000       1.0000000       1.0000000       6.0000000
6.0000000       1.0000000       0.8750000       0.9333333       8.0000000
7.0000000       1.0000000       1.0000000       1.0000000       5.0000000
8.0000000       1.0000000       1.0000000       1.0000000       2.0000000
9.0000000       0.8888889       1.0000000       0.9411765       8.0000000
10.0000000      1.0000000       1.0000000       1.0000000       8.0000000

Iter:400, training loss:0.08742103541529415, training accuracy:96.875
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       6.0000000
2.0000000       0.8000000       1.0000000       0.8888889       8.0000000
3.0000000       1.0000000       0.8333333       0.9090909       6.0000000
4.0000000       1.0000000       1.0000000       1.0000000       4.0000000
5.0000000       1.0000000       1.0000000       1.0000000       4.0000000
6.0000000       1.0000000       1.0000000       1.0000000       6.0000000
7.0000000       1.0000000       1.0000000       1.0000000       7.0000000
8.0000000       1.0000000       1.0000000       1.0000000       6.0000000
9.0000000       1.0000000       1.0000000       1.0000000       4.0000000
10.0000000      1.0000000       0.9230769       0.9600000       13.0000000

Iter:500, training loss:0.05873836245880005, training accuracy:98.4375
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       3.0000000
2.0000000       1.0000000       1.0000000       1.0000000       5.0000000
3.0000000       1.0000000       1.0000000       1.0000000       6.0000000
4.0000000       1.0000000       1.0000000       1.0000000       9.0000000
5.0000000       1.0000000       1.0000000       1.0000000       4.0000000
6.0000000       1.0000000       0.8571429       0.9230769       7.0000000
7.0000000       0.8571429       1.0000000       0.9230769       6.0000000
8.0000000       1.0000000       1.0000000       1.0000000       9.0000000
9.0000000       1.0000000       1.0000000       1.0000000       10.0000000
10.0000000      1.0000000       1.0000000       1.0000000       5.0000000

Iter:500, validation loss:260.1580978627665, validation accuracy:96.43954918032787
Iter:600, training loss:0.07584116043829209, training accuracy:98.4375
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       8.0000000
2.0000000       1.0000000       1.0000000       1.0000000       4.0000000
3.0000000       1.0000000       1.0000000       1.0000000       4.0000000
4.0000000       1.0000000       1.0000000       1.0000000       4.0000000
5.0000000       1.0000000       1.0000000       1.0000000       5.0000000
6.0000000       1.0000000       1.0000000       1.0000000       8.0000000
7.0000000       1.0000000       1.0000000       1.0000000       8.0000000
8.0000000       1.0000000       0.9230769       0.9600000       13.0000000
9.0000000       1.0000000       1.0000000       1.0000000       5.0000000
10.0000000      0.8333333       1.0000000       0.9090909       5.0000000

Iter:700, training loss:0.07973166944626336, training accuracy:98.4375
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       5.0000000
2.0000000       1.0000000       1.0000000       1.0000000       4.0000000
3.0000000       1.0000000       1.0000000       1.0000000       6.0000000
4.0000000       1.0000000       1.0000000       1.0000000       4.0000000
5.0000000       1.0000000       1.0000000       1.0000000       5.0000000
6.0000000       1.0000000       1.0000000       1.0000000       6.0000000
7.0000000       1.0000000       1.0000000       1.0000000       10.0000000
8.0000000       0.8000000       1.0000000       0.8888889       4.0000000
9.0000000       1.0000000       1.0000000       1.0000000       8.0000000
10.0000000      1.0000000       0.9166667       0.9565217       12.0000000

Iter:800, training loss:0.0063778595034221855, training accuracy:100.0
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       9.0000000
2.0000000       1.0000000       1.0000000       1.0000000       6.0000000
3.0000000       1.0000000       1.0000000       1.0000000       7.0000000
4.0000000       1.0000000       1.0000000       1.0000000       7.0000000
5.0000000       1.0000000       1.0000000       1.0000000       4.0000000
6.0000000       1.0000000       1.0000000       1.0000000       9.0000000
7.0000000       1.0000000       1.0000000       1.0000000       6.0000000
8.0000000       1.0000000       1.0000000       1.0000000       8.0000000
9.0000000       1.0000000       1.0000000       1.0000000       2.0000000
10.0000000      1.0000000       1.0000000       1.0000000       6.0000000

Iter:900, training loss:0.019673112167879484, training accuracy:100.0
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       3.0000000
2.0000000       1.0000000       1.0000000       1.0000000       4.0000000
3.0000000       1.0000000       1.0000000       1.0000000       3.0000000
4.0000000       1.0000000       1.0000000       1.0000000       5.0000000
5.0000000       1.0000000       1.0000000       1.0000000       6.0000000
6.0000000       1.0000000       1.0000000       1.0000000       10.0000000
7.0000000       1.0000000       1.0000000       1.0000000       7.0000000
8.0000000       1.0000000       1.0000000       1.0000000       7.0000000
9.0000000       1.0000000       1.0000000       1.0000000       12.0000000
10.0000000      1.0000000       1.0000000       1.0000000       7.0000000

Iter:1000, training loss:0.06137978002508307, training accuracy:96.875
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       5.0000000
2.0000000       1.0000000       1.0000000       1.0000000       7.0000000
3.0000000       1.0000000       1.0000000       1.0000000       8.0000000
4.0000000       0.8333333       0.8333333       0.8333333       6.0000000
5.0000000       1.0000000       1.0000000       1.0000000       5.0000000
6.0000000       1.0000000       1.0000000       1.0000000       10.0000000
7.0000000       1.0000000       1.0000000       1.0000000       3.0000000
8.0000000       0.8888889       0.8888889       0.8888889       9.0000000
9.0000000       1.0000000       1.0000000       1.0000000       7.0000000
10.0000000      1.0000000       1.0000000       1.0000000       4.0000000

Iter:1000, validation loss:238.62301345198944, validation accuracy:97.02868852459017
Iter:1100, training loss:0.023325103696013115, training accuracy:100.0
class           precision       recall          f1-score        num_true_labels
1.0000000       1.0000000       1.0000000       1.0000000       4.0000000
2.0000000       1.0000000       1.0000000       1.0000000       10.0000000
3.0000000       1.0000000       1.0000000       1.0000000       6.0000000
4.0000000       1.0000000       1.0000000       1.0000000       4.0000000
5.0000000       1.0000000       1.0000000       1.0000000       2.0000000
6.0000000       1.0000000       1.0000000       1.0000000       10.0000000
7.0000000       1.0000000       1.0000000       1.0000000       7.0000000
8.0000000       1.0000000       1.0000000       1.0000000       6.0000000
9.0000000       1.0000000       1.0000000       1.0000000       9.0000000
10.0000000      1.0000000       1.0000000       1.0000000       6.0000000
...
```
