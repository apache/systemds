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


# Layers supported in Caffe2DML

Caffe2DML to be as compatible with [the Caffe specification](http://caffe.berkeleyvision.org/tutorial/layers.html) as possible.
The main differences are given below along with the usage guide that mirrors the Caffe specification.

## Vision Layers

### Convolution Layer

Invokes [nn/layers/conv2d_builtin.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/conv2d_builtin.dml)
or [nn/layers/conv2d_depthwise.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/conv2d_depthwise.dml) layer.

**Required Parameters:**

- num_output: the number of filters
- kernel_size (or kernel_h and kernel_w): specifies height and width of each filter

**Optional Parameters:**

- bias_term (default true): specifies whether to learn and apply a set of additive biases to the filter outputs
- pad (or pad_h and pad_w) (default 0): specifies the number of pixels to (implicitly) add to each side of the input
- stride (or stride_h and stride_w) (default 1): specifies the intervals at which to apply the filters to the input
- group (g) (default 1): If g > 1, we restrict the connectivity of each filter to a subset of the input. 
Specifically, the input and output channels are separated into g groups, 
and the ith output group channels will be only connected to the ith input group channels.
Note: we only support depthwise convolution, hence `g` should be divisible by number of channels 

**Parameters that are ignored:**

- weight_filler: We use the heuristic by He et al., which limits the magnification of inputs/gradients 
during forward/backward passes by scaling unit-Gaussian weights by a factor of sqrt(2/n), 
under the assumption of relu neurons.
- bias_filler: We use `constant bias_filler` with `value:0`

**Sample Usage:**
```
layer {
    name: "conv1"
    type: "Convolution"
    bottom: "data"
    top: "conv1"
    # learning rate and decay multipliers for the filters
    param { lr_mult: 1 decay_mult: 1 }
    # learning rate and decay multipliers for the biases
    param { lr_mult: 2 decay_mult: 0 }
    convolution_param {
      num_output: 96     # learn 96 filters
      kernel_size: 11    # each filter is 11x11
      stride: 4          # step 4 pixels between each filter application
      weight_filler {
        type: "xavier" # initialize the filters from a Gaussian
      }
      bias_filler {
        type: "constant" # initialize the biases to zero (0)
        value: 0
      }
    }
  }
 ```
 
### Pooling Layer

Invokes [nn/layers/max_pool2d_builtin.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/max_pool2d_builtin.dml) layer.
 
**Required Parameters:**

- kernel_size (or kernel_h and kernel_w): specifies height and width of each filter

**Optional Parameters:**
- pool (default MAX): the pooling method. Currently, we only support MAX, not AVE, or STOCHASTIC.
- pad (or pad_h and pad_w) (default 0): specifies the number of pixels to (implicitly) add to each side of the input
- stride (or stride_h and stride_w) (default 1): specifies the intervals at which to apply the filters to the input

**Sample Usage:**
```
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3 # pool over a 3x3 region
    stride: 2      # step two pixels (in the bottom blob) between pooling regions
  }
}
```

### Deconvolution Layer

Invokes [nn/layers/conv2d_transpose.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/conv2d_transpose.dml)
or [nn/layers/conv2d_transpose_depthwise.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/conv2d_transpose_depthwise.dml) layer.

**Required Parameters:**

- num_output: the number of filters
- kernel_size (or kernel_h and kernel_w): specifies height and width of each filter

**Optional Parameters:**

- bias_term (default true): specifies whether to learn and apply a set of additive biases to the filter outputs
- pad (or pad_h and pad_w) (default 0): specifies the number of pixels to (implicitly) add to each side of the input
- stride (or stride_h and stride_w) (default 1): specifies the intervals at which to apply the filters to the input
- group (g) (default 1): If g > 1, we restrict the connectivity of each filter to a subset of the input. 
Specifically, the input and output channels are separated into g groups, 
and the ith output group channels will be only connected to the ith input group channels.
Note: we only support depthwise convolution, hence `g` should be divisible by number of channels 

**Parameters that are ignored:**

- weight_filler: We use the heuristic by He et al., which limits the magnification of inputs/gradients 
during forward/backward passes by scaling unit-Gaussian weights by a factor of sqrt(2/n), 
under the assumption of relu neurons.
- bias_filler: We use `constant bias_filler` with `value:0`

**Sample Usage:**
```
layer {
  name: "upconv_d5c_u4a"
  type: "Deconvolution"
  bottom: "u5d"
  top: "u4a"
  param {
    lr_mult: 0.0
    decay_mult: 0.0
  }
  convolution_param {
    num_output: 190
    bias_term: false
    pad: 1
    kernel_size: 4
    group: 190
    stride: 2
    weight_filler {
      type: "bilinear"
    }
  }
}
```


## Common Layers

### Inner Product / Fully Connected Layer

Invokes [nn/layers/affine.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/affine.dml) layer.

**Required Parameters:**

- num_output: the number of filters

**Parameters that are ignored:**
- weight_filler (default type: 'constant' value: 0): We use the heuristic by He et al., which limits the magnification
of inputs/gradients during forward/backward passes by scaling unit-Gaussian weights by a factor of sqrt(2/n), under the
assumption of relu neurons.
- bias_filler (default type: 'constant' value: 0): We use the default type and value.
- bias_term (default true): specifies whether to learn and apply a set of additive biases to the filter outputs. We use `bias_term=true`.

**Sample Usage:**
```
layer {
  name: "fc8"
  type: "InnerProduct"
  # learning rate and decay multipliers for the weights
  param { lr_mult: 1 decay_mult: 1 }
  # learning rate and decay multipliers for the biases
  param { lr_mult: 2 decay_mult: 0 }
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
  bottom: "fc7"
  top: "fc8"
}
```

### Dropout Layer

Invokes [nn/layers/dropout.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/dropout.dml) layer.

**Optional Parameters:**

- dropout_ratio(default = 0.5): dropout ratio

**Sample Usage:**
```
layer {
  name: "drop1"
  type: "Dropout"
  bottom: "relu3"
  top: "drop1"
  dropout_param {
    dropout_ratio: 0.5
  }
}
```

## Normalization Layers

### BatchNorm Layer

This is used in combination with Scale layer.

Invokes [nn/layers/batch_norm2d.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/batch_norm2d.dml) layer.

**Optional Parameters:**
- moving_average_fraction (default = .999): Momentum value for moving averages. Typical values are in the range of [0.9, 0.999].
- eps (default = 1e-5): Smoothing term to avoid divide by zero errors. Typical values are in the range of [1e-5, 1e-3].

**Parameters that are ignored:**
- use_global_stats: If false, normalization is performed over the current mini-batch 
and global statistics are accumulated (but not yet used) by a moving average.
If true, those accumulated mean and variance values are used for the normalization.
By default, it is set to false when the network is in the training phase and true when the network is in the testing phase.

**Sample Usage:**
```
layer {
	bottom: "conv1"
	top: "conv1"
	name: "bn_conv1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: true
	}
}
layer {
	bottom: "conv1"
	top: "conv1"
	name: "scale_conv1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}
```

## Activation / Neuron Layers

In general, activation / Neuron layers are element-wise operators, taking one bottom blob and producing one top blob of the same size. 
In the layers below, we will ignore the input and out sizes as they are identical.

### ReLU / Rectified-Linear Layer

Invokes [nn/layers/relu.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/relu.dml) layer.

**Parameters that are ignored:**
- negative_slope (default 0): specifies whether to leak the negative part by multiplying it with the slope value rather than setting it to 0.

**Sample Usage:**
```
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
```

### TanH Layer

Invokes [nn/layers/tanh.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/tanh.dml) layer.

**Sample Usage:**
```
layer {
  name: "tanh1"
  type: "TanH"
  bottom: "conv1"
  top: "conv1"
}
```

### Sigmoid Layer

Invokes [nn/layers/sigmoid.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/sigmoid.dml) layer.

**Sample Usage:**
```
layer {
  name: "sigmoid1"
  type: "Sigmoid"
  bottom: "conv1"
  top: "conv1"
}
```


### Threshold Layer

Computes `X > threshold`

**Parameters that are ignored:**
- threshold (default: 0):Strictly positive values

**Sample Usage:**
```
layer {
  name: "threshold1"
  type: "Threshold"
  bottom: "conv1"
  top: "conv1"
}
```

## Utility Layers

### Eltwise Layer

Element-wise operations such as product or sum between two blobs.

**Parameters that are ignored:**
- operation(default: SUM): element-wise operation. only SUM supported for now.
- table_prod_grad(default: true): Whether to use an asymptotically slower (for >2 inputs) but stabler method
of computing the gradient for the PROD operation. (No effect for SUM op.)

**Sample Usage:**
```
layer {
	bottom: "res2a_branch1"
	bottom: "res2a_branch2c"
	top: "res2a"
	name: "res2a"
	type: "Eltwise"
}
```

### Concat Layer

**Inputs:**
- `n_i * c_i * h * w` for each input blob i from 1 to K.

**Outputs:**
- out: Outputs, of shape
  - if axis = 0: `(n_1 + n_2 + ... + n_K) * c_1 * h * w`, and all input `c_i` should be the same.
  - if axis = 1: `n_1 * (c_1 + c_2 + ... + c_K) * h * w`, and all input `n_i` should be the same.

**Optional Parameters:**
- axis (default: 1): The axis along which to concatenate.

**Sample Usage:**
```
layer {
  name: "concat_d5cc_u5a-b"
  type: "Concat"
  bottom: "u5a"
  bottom: "d5c"
  top: "u5b"
}
```

### Softmax Layer

Invokes [nn/layers/softmax.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/softmax.dml) layer.

Computes the forward pass for a softmax classifier.  The inputs
are interpreted as unnormalized, log-probabilities for each of
N examples, and the softmax function transforms them to normalized
probabilities.

This can be interpreted as a generalization of the sigmoid
function to multiple classes.

`probs_ij = e^scores_ij / sum(e^scores_i)`

**Parameters that are ignored:**
- axis (default: 1): The axis along which to perform the softmax.

**Sample Usage:**
```
layer {
  name: "sm"
  type: "Softmax"
  bottom: "score"
  top: "sm"
}
```

## Loss Layers

Loss drives learning by comparing an output to a target and assigning cost to minimize. 
The loss itself is computed by the forward pass and the gradient w.r.t. to the loss is computed by the backward pass.

### Softmax with Loss Layer

The softmax loss layer computes the multinomial logistic loss of the softmax of its inputs. 
It’s conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient.

Invokes [nn/layers/softmax.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/softmax.dml)
and [nn/layers/cross_entropy_loss.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/cross_entropy_loss.dml) 
for classification problems.

For image segmentation problems, invokes [nn/layers/softmax2d_loss.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/softmax2d_loss.dml) layer.

**Sample Usage:**
```
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
```

### Euclidean layer

The Euclidean loss layer computes the sum of squares of differences of its two inputs.

Invokes [nn/layers/l2_loss.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/l2_loss.dml) layer.

**Sample Usage:**
```
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
```


# Frequently asked questions

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

Again, the SystemML engine doesnot invoke (or depend on) Caffe for any of its runtime operators.
Since the grammar files for the respective APIs (i.e. `caffe.proto`) are used by SystemML, 
we include their licenses in our jar files.

#### How can I speedup the training with Caffe2DML ?

- Enable native BLAS to improve the performance of CP convolution and matrix multiplication operators.
If you are using OpenBLAS, please ensure that it was built with `USE_OPENMP` flag turned on.
For more detail see http://apache.github.io/systemml/native-backend

```python
caffe2dmlObject.setConfigProperty("sysml.native.blas", "auto")
```

- Turn on the experimental codegen feature. This should help reduce unnecessary allocation cost after every binary operation.

```python
caffe2dmlObject.setConfigProperty("sysml.codegen.enabled", "true").setConfigProperty("sysml.codegen.plancache", "true")
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

