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

## Table of Contents

  * [Introduction](#introduction)
  * [Built-In Construction Functions](#built-in-construction-functions)
    * [`tensor`-Function](#tensor-function)
  * [DML-Bodied Built-In functions](#dml-bodied-built-in-functions)
    * [`lm`-Function](#lm-function)
    * [`lmDS`-Function](#lmds-function)
    * [`lmCG`-Function](#lmcg-function)
    * [`lmpredict`-Function](#lmpredict-function)
    * [`steplm`-Function](#steplm-function)
    * [`slicefinder`-Function](#slicefinder-function)
    
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
| data    | Matrix[?], Tensor[?], Scalar[?] | required | The data with which the tensor should be filled. See [`data`-Argument](#data-argument).|
| dims    | Matrix[Integer], Tensor[Integer], Scalar[String], List[Integer] | required | The dimensions of the tensor. See [`dims`-Argument](#dims-argument). |
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

Note that reshape construction is not yet supported for **SPARK** execution.

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

## `steplm`-Function

The `steplm`-function (stepwise linear regression) implements a classical forward feature selection method.
This method iteratively runs what-if scenarios and greedily selects the next best feature until the Akaike 
information criterion (AIC) does not improve anymore. Each configuration trains a regression model via `lm`,
which in turn calls either the closed form `lmDS` or iterative `lmGC`.

### Usage
```r
steplm(X, y, icpt);
```

### Arguments
| Name    | Type           | Default  | Description |
| :------ | :------------- | -------- | :---------- |
| X       | Matrix[Double] | required | Matrix of feature vectors. |
| y       | Matrix[Double] | required | 1-column matrix of response values. |
| icpt    | Integer        | `0`      | Intercept presence, shifting and rescaling the columns of X ([Details](#icpt-argument))|
| reg     | Double         | `1e-7`   | Regularization constant (lambda) for L2-regularization. set to nonzero for highly dependent/sparse/numerous features|
| tol     | Double         | `1e-7`   | Tolerance (epsilon); conjugate gradient procedure terminates early if L2 norm of the beta-residual is less than tolerance * its initial norm|
| maxi    | Integer        | `0`      | Maximum number of conjugate gradient iterations. 0 = no maximum |
| verbose | Boolean        | `TRUE`   | If `TRUE` print messages are activated |

### Returns
| Type           | Description |
| :------------- | :---------- |
| Matrix[Double] | Matrix of regression parameters (the betas) and its size depend on `icpt` input value. (C in the example)|
| Matrix[Double] | Matrix of `selected` features ordered as computed by the algorithm. (S in the example)|

##### `icpt`-Argument

The *icpt-arg* can be set to 2 modes:
 
  * 0 = no intercept, no shifting, no rescaling
  * 1 = add intercept, but neither shift nor rescale X

##### `selected`-Output

If the best AIC is achieved without any features the matrix of *selected* features contains 0. Moreover, in this case no further statistics will be produced 

### Example
```r
X = rand (rows = 50, cols = 10)
y = X %*% rand(rows=ncol(X), 1)
[C, S] = steplm(X = X, y = y, icpt = 1);
`

## `slicefinder`-Function

The `slicefinder`-function returns top-k worst performing subsets according to a model calculation.

### Usage
```r
slicefinder(X,W, y, k, paq, S);
```

### Arguments
| Name    | Type           | Default  | Description |
| :------ | :------------- | -------- | :---------- |
| X       | Matrix[Double] | required | Recoded dataset into Matrix |
| W       | Matrix[Double] | required | Trained model |
| y       | Matrix[Double] | required | 1-column matrix of response values. |
| k       | Integer        | 1        | Number of subsets required |
| paq     | Integer        | 1        | amount of values wanted for each col, if paq = 1 then its off |
| S       | Integer        | 2        | amount of subsets to combine (for now supported only 1 and 2) |

### Returns
| Type           | Description |
| :------------- | :---------- |
| Matrix[Double] | Matrix containing the information of top_K slices (relative error, standart error, value0, value1, col_number(sort), rows, cols,range_row,range_cols, value00, value01,col_number2(sort), rows2, cols2,range_row2,range_cols2) |

### Usage
```r
X = rand (rows = 50, cols = 10)
y = X %*% rand(rows = ncol(X), cols = 1)
w = lm(X = X, y = y)
ress = slicefinder(X = X,W = w, Y = y,  k = 5, paq = 1, S = 2);
`

## `outlier-Function

An outlier is any value that is numerically distant from most of the other data points in a set of data.
This outlier function takes a matrix  data set as input from where it determines which number or numbers  has the largest diference from mean,
The number which has the largest diference from mean is indicated as an outlier.


### Usage
```r
outlier(X,opposite);
```

### Arguments
| Name    | Type           | Default  | Description |
| :------ | :------------- | -------- | :---------- |
| X       | Matrix[Double] | required | Matrix of Recoded dataset for outlier evaluation |
|opposite| Boolean | required | (1)TRUE for evaluating outlier from upper quartile range |
                                                       |(0)FALSE for evaluating outlier from lower quartile range|
### Returns
| Type           | Description |
| :------------- | :---------- |
| Matrix[Double] | matrix indicating outlier values |

### Example
```r
X = rand (rows = 50, cols = 10)
opposite = 1
outlier(X=X,opposite=opposite)
```

## outlierByIQR - Function

Builtin function for detecting and repairing outliers using Interquartile Range.
A commonly used rule says that a data point is an outlier if it is more than 1.5 IQR
above the third quartile or below the first quartile.
outlierByIQR function computes the matrix and set's a lower-bound quartile range and upper-bound quartile range 
and the number which is less then the lower-bound or higher then the upper-bound is treated as a outlier, hence
removed from the matrix.


### Usage

outlierByIQR(X,k,repair_method,max_iterations,verbose)

###  Arguments

| Name    | Type           | Default  | Description |
| :------ | :------------- | -------- | :---------- |
| X       | Matrix[Double] | required | matrix with outliers |
|k         |     Double 	   |  1.5         | a constant used to discern outliers k*IQR 
 |isIterative|  Boolean | TRUE   |iterative repair or single repair 
 |repairMethod|   Integer|  1           | values: 0 = delete rows having outliers, 
                                                              1 = replace outliers with zeros 
                                            		      2 = replace outliers as missing values 
 |max_iterations|  Integer | 0      | values: 0 = arbitrary number of iteraition until all outliers are removed, 
                                                            n = any constant defined by user
###  Returns

| Type           | Description |
| :------------- | :---------- |
| Matrix[Double] | matrix without any outlier. |

###  Example

`X = rand (rows=10,cols=10)
Z = outlierByIQR(X=X,k=1.5,repairMethod=0,max_iterations=3,verbose=1)
print("\n"+toString(Z))


##outlierBySd - function

Builtin function for detecting and repairing outliers using standard deviation.
Acording to three sigma rule if a value falls outside of three times the standard deviations then it is an outlier value.
In this function outlierBySd a matrix of trained data sets is provided from which it computes the upper-bound and lower-bound of data
and any value that is more then upper-bound or lower then lower-bound is treated as an outlier and then gets filtered from the data set.

###  usage

outlierBySd(X,k,repairMethod,max_iterations,verbose)

###  Arguments

| Name    | Type           | Default  | Description |
| :------ | :------------- | -------- | :---------- |
| X         |      Double    |---       |Matrix with outlier values |
|k            |   Double    |3        | threshold values 1, 2, 3 for 68%, 95%, 99.7% respectively (3-sigma rule)
|repairMethod|    Integer  | 1 |        values: 0 = delete rows having outliers, 1 = replace outliers as  zeros 
                                                               2 = replace outliers as missing values 
| max_iterations|  Integer |   0  |       values: 0 = arbitrary number of iteration until all outliers are removed, 
                                                              n = any constant defined by user
### Returns

| Type           | Description |
| :------------- | :---------- |
| Matrix[Double] | matrix with no outlier |

### Example

X = rand (rows = 20, cols = 10)
Z=outlierBySd(X=X, k=3,repairMethod = 1,max_iterations = 10,verbose = 1)
print("\n"+toString(Z))



### confusionMatrix Function

A confusion matrix is a technique for summarizing the performance of a classification algorithm.
Calculating a confusion matrix can give you a better idea of what your classification model is getting right and what types of errors it is making.
This confusionMatrix function accepts two matrices with one column each, these two matrices are vector for prediction and one-hot-encoded matrix respectively.
Then it computes the max value of each vector and compare them, after whichit calculates and returns the sum of classifications and the average of each true class.

### Usage

confusionMatrix(P,Y)

### Arguments

| Name    | Type                        | Default  | Description |
| :------ | :-------------                    | -------- | :---------- |
| P         |      Matrix[Double]    |---       |vector of prediction |
| Y         |      Matrix[Double]    |---       | vector of Golden standard One Hot Encoded|

### Returns
 
|Name  		| Type           | Description |
|:-----------------| :------------- | :---------- |
|ConfusionSum| Matrix[Double] | The Confusion Matrix Sums of classifications |
|ConfusionAvg | Matrix[Double] | The Confusion Matrix averages of each true class|

### Example

 #here numClasses is assigned to 1 as numClasses is directly proportional to the 
#number of columns in the one hot data matrix, as confusion matrix accepts only matrices with one column.

numClasses = 1  
z = rand(rows=5,cols=1,min = 1 , max = 9)
X = round(rand(rows = 5, cols = 1, min = 1, max = numClasses))
y = toOneHot(X,numClasses)
print("\nOne-HOT\n"+toString(y)+"\nprediction matrix:\n"+toString(z))
[sum,avg] = confusionMatrix(P=z,Y=y)
print("\nconfusion-matrix-sum\n"+toString(sum)+"\nconfusion-matrix-avg\n"+toString(avg))
