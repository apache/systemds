<!--
{% comment %}
Modifications Copyright 2019 Graz University of Technology

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

## Table of Contents

  * [Introduction](#introduction)
  * [Built-In Construction Functions](#built-in-construction-functions)
    * [`tensor`-Function](#tensor-function)
  * [DML-Bodied Built-In functions](#dml-bodied-built-in-functions)
    * [`lm`-Function](#lm-function)
    * [`lmDS`-Function](#lmds-function)
    * [`lmCG`-Function](#lmcg-function)
    * [`lmpredict`-Function](#lmpredict-function)
    
# Introduction

The DML (Declarative Machine Learning) language has built-in functions which enable access to both low- and high-level functions
to support all kinds of use cases.

Builtins are either implemented on a compiler level or as DML scripts that are loaded at compile time.

# Built-In Construction Functions

There are some functions which generate an object for us. They create matrices, tensors, lists and other non-primitive
objects.

## `tensor`-Function

The `tensor`-function creates a **tensor** for us.

### Usage
```r
tensor(data, dims, byRow = TRUE)
```

### Arguments
| Name    | Type           | Default  | Description |
| :------ | :------------- | -------- | :---------- |
| data    | Matrix[?], Tensor[?], String, Scalar[?] | required | The data with which the tensor should be filled. See [`data`-Argument](#data-argument).|
| dims    | Matrix[Integer], Tensor[Integer], String, List[Integer] | required | The dimensions of the tensor. See [`dims`-Argument](#dims-argument). |
| byRow   | Boolean        | TRUE     | NOT USED. Will probably be removed or replaced. |

Note that this function is highly **unstable** and will be overworked and might change signature and functionality.

### Returns
| Type           | Description |
| :------------- | :---------- |
| Tensor[?] | The generated Tensor. Will support more datatypes than `Double`. |

##### `data`-Argument

The `data`-argument can be a `Matrix` of any datatype from which the elements will be taken and placed in the tensor 
until filled. If given as a `Tensor` the same procedure takes place. We iterate through `Matrix` and `Tensor` by starting
with each dimension index at `0` and then incrementing the lowest one, until we made a complete pass over the dimension,
and then increasing the dimension index above. This will be done until the `Tensor` is completely filled.

If `data` is a `Scalar`, we fill the whole tensor with the value.

##### `dims`-Argument

The dimension of the tensor can either be given by a vector represented by either by a `Matrix`, `Tensor`, `String` or `List`.
Dimensions given by a `String` will be expected to be concatenated by spaces.

### Example
```r
print("Dimension matrix:");
d = matrix("2 3 4", 1, 3);
print(toString(d, decimal=1))

print("Tensor A: Fillvalue=3, dims=2 3 4");
A = tensor(3, d); # fill with value, dimensions given by matrix
print(toString(A))

print("Tensor B: Reshape A, dims=4 2 3");
B = tensor(A, "4 2 3"); # reshape tensor, dimensions given by string
print(toString(B))

print("Tensor C: Reshape dimension matrix, dims=1 3");
C = tensor(d, list(1, 3)); # values given by matrix, dimensions given by list
print(toString(C, decimal=1))

print("Tensor D: Values=tst, dims=Tensor C");
D = tensor("tst", C); # fill with string, dimensions given by tensor
print(toString(D))
```

# DML-Bodied Built-In Functions

**DML-bodied built-in functions** are written as DML-Scripts and executed as such when called.

## `lm`-Function

The `lm`-function solves linear regression using either the **direct solve method** or the **conjugate gradient algorithm**
depending on the input size of the matrices (See [`lmDS`-function](#lmds-function) and 
[`lmCG`-function](#lmcg-function) respectively).

### Usage
```r
lm(X, y, icpt = 0, reg = 1e-7, tol = 1e-7, maxi = 0, verbose = TRUE)
```

### Arguments
| Name    | Type           | Default  | Description |
| :------ | :------------- | -------- | :---------- |
| X       | Matrix[Double] | required | Matrix of feature vectors. |
| y       | Matrix[Double] | required | 1-column matrix of response values. |
| icpt    | Integer        | `0`      | Intercept presence, shifting and rescaling the columns of X ([Details](#icpt-argument))|
| reg     | Double         | `1e-7`   | Regularization constant (lambda) for L2-regularization. set to nonzero for highly dependant/sparse/numerous features|
| tol     | Double         | `1e-7`   | Tolerance (epsilon); conjugate gradient procedure terminates early if L2 norm of the beta-residual is less than tolerance * its initial norm|
| maxi    | Integer        | `0`      | Maximum number of conjugate gradient iterations. 0 = no maximum |
| verbose | Boolean        | `TRUE`   | If `TRUE` print messages are activated |

Note that if number of *features* is small enough (`rows of X/y < 2000`), the [`lmDS`-Function'](#lmds-function)
is called internally and parameters `tol` and `maxi` are ignored.

### Returns
| Type           | Description |
| :------------- | :---------- |
| Matrix[Double] | 1-column matrix of weights. |

##### `icpt`-Argument

The *icpt-argument* can be set to 3 modes:
 
  * 0 = no intercept, no shifting, no rescaling
  * 1 = add intercept, but neither shift nor rescale X
  * 2 = add intercept, shift & rescale X columns to mean = 0, variance = 1

### Example
```r
X = rand (rows = 50, cols = 10)
y = X %*% rand(rows=ncol(X), 1)
lm(X = X, y = y)
```

## `lmDS`-Function

The `lmDS`-function solves linear regression by directly solving the *linear system*.

### Usage
```r
lmDS(X, y, icpt = 0, reg = 1e-7, verbose = TRUE)
```

### Arguments
| Name    | Type           | Default  | Description |
| :------ | :------------- | -------- | :---------- |
| X       | Matrix[Double] | required | Matrix of feature vectors. |
| y       | Matrix[Double] | required | 1-column matrix of response values. |
| icpt    | Integer        | `0`      | Intercept presence, shifting and rescaling the columns of X ([Details](#icpt-argument))|
| reg     | Double         | `1e-7`   | Regularization constant (lambda) for L2-regularization. set to nonzero for highly dependant/sparse/numerous features|
| verbose | Boolean        | `TRUE`   | If `TRUE` print messages are activated |

### Returns
| Type           | Description |
| :------------- | :---------- |
| Matrix[Double] | 1-column matrix of weights. |

### Example
```r
X = rand (rows = 50, cols = 10)
y = X %*% rand(rows=ncol(X), 1)
lmDS(X = X, y = y)
```

## `lmCG`-Function

The `lmCG`-function solves linear regression using the *conjugate gradient algorithm*.

### Usage
```r
lmCG(X, y, icpt = 0, reg = 1e-7, tol = 1e-7, maxi = 0, verbose = TRUE)
```

### Arguments
| Name    | Type           | Default  | Description |
| :------ | :------------- | -------- | :---------- |
| X       | Matrix[Double] | required | Matrix of feature vectors. |
| y       | Matrix[Double] | required | 1-column matrix of response values. |
| icpt    | Integer        | `0`      | Intercept presence, shifting and rescaling the columns of X ([Details](#icpt-argument))|
| reg     | Double         | `1e-7`   | Regularization constant (lambda) for L2-regularization. set to nonzero for highly dependant/sparse/numerous features|
| tol     | Double         | `1e-7`   | Tolerance (epsilon); conjugate gradient procedure terminates early if L2 norm of the beta-residual is less than tolerance * its initial norm|
| maxi    | Integer        | `0`      | Maximum number of conjugate gradient iterations. 0 = no maximum |
| verbose | Boolean        | `TRUE`   | If `TRUE` print messages are activated |

### Returns
| Type           | Description |
| :------------- | :---------- |
| Matrix[Double] | 1-column matrix of weights. |

### Example
```r
X = rand (rows = 50, cols = 10)
y = X %*% rand(rows=ncol(X), 1)
lmCG(X = X, y = y, maxi = 10)
```

## `lmpredict`-Function

The `lmpredict`-function predicts the class of a feature vector.

### Usage
```r
lmpredict(X, w)
```

### Arguments
| Name    | Type           | Default  | Description |
| :------ | :------------- | -------- | :---------- |
| X       | Matrix[Double] | required | Matrix of feature vector(s). |
| w       | Matrix[Double] | required | 1-column matrix of weights. |
| icpt    | Matrix[Double] | `0`      | Intercept presence, shifting and rescaling of X ([Details](#icpt-argument))|

### Returns
| Type           | Description |
| :------------- | :---------- |
| Matrix[Double] | 1-column matrix of classes. |

### Example
```r
X = rand (rows = 50, cols = 10)
y = X %*% rand(rows=ncol(X), 1)
w = lm(X = X, y = y)
yp = lmpredict(X, w)
```

