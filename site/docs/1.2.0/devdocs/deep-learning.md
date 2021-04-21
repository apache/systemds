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

# Initial prototype for Deep Learning

## Representing tensor and images in SystemML

In this prototype, we represent a tensor as a matrix stored in a row-major format,
where first dimension of tensor and matrix are exactly the same. For example, a tensor (with all zeros)
of shape [3, 2, 4, 5] can be instantiated by following DML statement:
```sh
A = matrix(0, rows=3, cols=2*4*5) 
```
### Tensor functions:

#### Element-wise arithmetic operators:
Following operators work out-of-the box when both tensors X and Y have same shape:

* Element-wise exponentiation: `X ^ Y`
* Element-wise unary minus: `-X`
* Element-wise integer division: `X %/% Y`
* Element-wise modulus operation: `X %% Y`
* Element-wise multiplication: `X * Y`
* Element-wise division: `X / Y`
* Element-wise addition: `X + Y`
* Element-wise subtraction: `X - Y`

SystemML does not support implicit broadcast for above tensor operations, however one can write a DML-bodied function to do so.
For example: to perform the above operations with broadcasting on second dimensions, one can use the below `rep(Z, n)` function:
``` python
rep = function(matrix[double] Z, int C) return (matrix[double] ret) {
	ret = Z
	for(i in 2:C) {
		ret = cbind(ret, Z)
	}
}
```
Using the above `rep(Z, n)` function, we can realize the element-wise arithmetic operation with broadcasting. Here are some examples:
* X of shape [N, C, H, W] and Y of shape [1, C, H, W]: `X + Y` (Note: SystemML does implicit broadcasting in this case because of the way 
it represents the tensor)
* X of shape [1, C, H, W] and Y of shape [N, C, H, W]: `X + Y` (Note: SystemML does implicit broadcasting in this case because of the way 
it represents the tensor)
* X of shape [N, C, H, W] and Y of shape [N, 1, H, W]: `X + rep(Y, C)`
* X of shape [N, C, H, W] and Y of shape [1, 1, H, W]: `X + rep(Y, C)`
* X of shape [N, 1, H, W] and Y of shape [N, C, H, W]: `rep(X, C) + Y`
* X of shape [1, 1, H, W] and Y of shape [N, C, H, W]: `rep(X, C) + Y`

TODO: Map the NumPy tensor calls to DML expressions.

## Representing images in SystemML

The images are assumed to be stored NCHW format, where N = batch size, C = #channels, H = height of image and W = width of image. 
Hence, the images are internally represented as a matrix with dimension (N, C * H * W).

## Convolution and Pooling built-in functions

This prototype also contains initial implementation of forward/backward functions for 2D convolution and pooling:
* `conv2d(x, w, ...)`
* `conv2d_backward_filter(x, dout, ...)` and `conv2d_backward_data(w, dout, ...)`
* `max_pool(x, ...)` and `max_pool_backward(x, dout, ...)`

The required arguments for all above functions are:
* stride=[stride_h, stride_w]
* padding=[pad_h, pad_w]
* input_shape=[numImages, numChannels, height_image, width_image]

The additional required argument for conv2d/conv2d_backward_filter/conv2d_backward_data functions is:
* filter_shape=[numFilters, numChannels, height_filter, width_filter]

The additional required argument for max_pool/avg_pool functions is:
* pool_size=[height_pool, width_pool]

The results of these functions are consistent with Nvidia's CuDNN library.

### Border mode:
* To perform valid padding, use `padding = (input_shape-filter_shape)*(stride-1)/ 2`. (Hint: for stride length of 1, `padding = [0, 0]` performs valid padding).

* To perform full padding, use `padding = ((stride-1)*input_shape + (stride+1)*filter_shape - 2*stride) / 2`. (Hint: for stride length of 1, `padding = [filter_h-1, filter_w-1]` performs full padding).

* To perform same padding, use `padding = (input_shape*(stride-1) + filter_shape - stride)/2`. (Hint: for stride length of 1, `padding = [(filter_h-1)/2, (filter_w-1)/2]` performs same padding).

### Explanation of backward functions for conv2d

Consider one-channel 3 X 3 image =
  
| x1 | x2 | x3 |
|----|----|----|
| x4 | x5 | x6 |
| x7 | x8 | x9 |

and one 2 X 2 filter:

| w1 | w2 |
|----|----|
| w3 | w4 |

Then, `conv2d(x, w, stride=[1, 1], padding=[0, 0], input_shape=[1, 1, 3, 3], filter_shape=[1, 1, 2, 2])` produces following tensor
of shape `[1, 1, 2, 2]`, which is represented as `1 X 4` matrix in NCHW format:

| `w1*x1 + w2*x2 + w3*x4 + w4*x5` | `w1*x2 + w2*x3 + w3*x5 + w4*x6` | `w1*x4 + w2*x5 + w3*x7 + w4*x8` | `w1*x5 + w2*x6 + w3*x8 + w4*x9` |
|---------------------------------|---------------------------------|---------------------------------|---------------------------------|


Let the error propagated from above layer is

| y1 | y2 | y3 | y4 |
|----|----|----|----|

Then `conv2d_backward_filter(x, y, stride=[1, 1], padding=[0, 0], input_shape=[1, 1, 3, 3], filter_shape=[1, 1, 2, 2])` produces following 
updates for the filter:

| `y1*x1 + y2*x2 + y3*x4 + y4*x5` | `y1*x2 + y2*x3 + y3*x5 + y4*x6` |
|---------------------------------|---------------------------------|
| `y1*x4 + y2*x5 + y3*x7 + y4*x8` | `y1*x5 + y2*x6 + y3*x8 + y4*x9` |

Note: since the above update is a tensor of shape [1, 1, 2, 2], it will be represented as matrix of dimension [1, 4].

Similarly, `conv2d_backward_data(w, y, stride=[1, 1], padding=[0, 0], input_shape=[1, 1, 3, 3], filter_shape=[1, 1, 2, 2])` produces following 
updates for the image:


| `w1*y1`         | `w2*y1 + w1*y2`                 | `w2*y2`         |
|-----------------|---------------------------------|-----------------|
| `w3*y1 + w1*y3` | `w4*y1 + w3*y2 + w2*y3 + w1*y4` | `w4*y2 + w2*y4` |
| `w3*y3`         | `w4*y3 + w3*y4`                 | `w4*y4`         |

# Caffe2DML examples

## Training using Caffe models on Lenet

The below script also demonstrates how to save the trained model.

```python
# Download the MNIST dataset
from mlxtend.data import mnist_data
import numpy as np
from sklearn.utils import shuffle
X, y = mnist_data()
X, y = shuffle(X, y)
num_classes = np.unique(y).shape[0]
img_shape = (1, 28, 28)

# Split the data into training and test
n_samples = len(X)
X_train = X[:int(.9 * n_samples)]
y_train = y[:int(.9 * n_samples)]
X_test = X[int(.9 * n_samples):]
y_test = y[int(.9 * n_samples):]

# Download the Lenet network
import urllib
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/lenet/mnist/lenet.proto', 'lenet.proto')
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/lenet/mnist/lenet_solver.proto', 'lenet_solver.proto')

# Train Lenet On MNIST using scikit-learn like API
from systemml.mllearn import Caffe2DML
lenet = Caffe2DML(sqlCtx, solver='lenet_solver.proto').set(max_iter=500, debug=True).setStatistics(True)
print('Lenet score: %f' % lenet.fit(X_train, y_train).score(X_test, y_test))

# Save the trained model
lenet.save('lenet_model')
```

## Load the trained model and retrain (i.e. finetuning)

```python
# Fine-tune the existing trained model
new_lenet = Caffe2DML(sqlCtx, solver='lenet_solver.proto', weights='lenet_model').set(max_iter=500, debug=True)
new_lenet.fit(X_train, y_train)
new_lenet.save('lenet_model')
```

## Perform prediction using the above trained model

```python
# Use the new model for prediction
predict_lenet = Caffe2DML(sqlCtx, solver='lenet_solver.proto', weights='lenet_model')
print('Lenet score: %f' % predict_lenet.score(X_test, y_test))
```

Similarly, you can perform prediction using the pre-trained ResNet network

```python
from systemml.mllearn import Caffe2DML
from pyspark.sql import SQLContext
import numpy as np
import urllib, os, scipy.ndimage
from PIL import Image
import systemml as sml

# ImageNet specific parameters
img_shape = (3, 224, 224)

# Downloads a jpg image, resizes it to 224 and return as numpy array in N X CHW format
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/MountainLion.jpg/312px-MountainLion.jpg'
outFile = 'test.jpg'
urllib.urlretrieve(url, outFile)
input_image = sml.convertImageToNumPyArr(Image.open(outFile), img_shape=img_shape)

# Download the ResNet network
import urllib
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/resnet/ilsvrc12/ResNet_50_network.proto', 'ResNet_50_network.proto')
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/resnet/ilsvrc12/ResNet_50_solver.proto', 'ResNet_50_solver.proto')

# Assumes that you have cloned the model_zoo repository
# git clone https://github.com/niketanpansare/model_zoo.git
resnet = Caffe2DML(sqlCtx, solver='ResNet_50_solver.proto', weights='~/model_zoo/caffe/vision/resnet/ilsvrc12/ResNet_50_pretrained_weights').set(input_shape=img_shape)
resnet.predict(input_image)
```