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

Invokes [nn/layers/tanh.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/sigmoid.dml) layer.

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

Invokes [nn/layers/softmax_loss.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/softmax_loss.dml) or
[nn/layers/softmax2d_loss.dml](https://github.com/apache/systemml/blob/master/scripts/nn/layers/softmax2d_loss.dml) layer.

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
