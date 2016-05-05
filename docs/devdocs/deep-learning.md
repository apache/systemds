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
