---
layout: site
title: Algorithms Reference Factorization Machines
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

### Factorization Description

The Factorization Machine (FM), is a general predictor like SVMs but is also
able to estimate reliable parameters under very high sparsity. The factorization machine
models all nested variable interactions (compared to a polynomial kernel in SVM), but
uses a factorized parameterization instead of a dense parameterisation like in SVMs.

## Core Model

### Model Equation

$$ \hat{y}(x) =
w_0 +
\sum_{i=1}^{n} w_i x_i +
\sum_{i=1}^{n} \sum_{j=i+1}^{n} \left <v_i, v_j \right > x_i x_j
$$

 where the model parameters that have to be estimated are:

 $$
 w_0 \in R,
 W   \in R^n,
 V   \in R^{n \times k}
 $$

and

 $$
   \left <\cdot, \cdot \right >
 $$

is the dot product of two vectors of size $k$:

 $$
 \left <v_i, v_j \right > = \sum_{f=1}^{k} v_{i,f} \cdot v_{j,f}
 $$

A row $v_i$ with in $V$ describes the $i$th variable with $k \in N_0^+$ 
factors. $k$ is a hyperparameter, that defines the dimensionality of
factorization.

- $ w_0 $ : global bias
- $ w_j $ : models the strength of the ith variable
- $ w_{i,j} = \left <v_i, v_j \right> $ : models the interaction between
the $i$th & $j$th variable.

Instead of using an own model parameter

$$ w_{i,j} \in R $$

for each interaction, the FM models the interaction by factorizing it.

### Expressiveness

It is well known that for any positive definite matrix $W$, there exists a 
matrix $V$ such that $W = V \cdot V^t$ provided that $k$ is sufficiently
large. This shows that an FM can express any interaction matrix $W$ if $k$
is chosen large enough.

### Parameter Estimation Under Sparsity

In sparse settings, there is usually not enough data to estimate interaction
between variables directly & independently. FMs can estimate interactions even
in these settings well because they break the independence of the interaction
parameters by factorizing them.

### Computation

Due to factorization of pairwise interactions, there is not model parameter
that directly depends on two variables ( e.g., a parameter with an index 
$(i,j)$ ). So, the pairwise interactions can be reformulated as shown below.

$$
\sum_{i=1}^n \sum_{j=i+1}^n \left <v_i, v_j \right > x_i x_j
$$

$$
= {1 \over 2} \sum_{i=1}^n \sum_{j=1}^n x_i x_j - {1 \over 2} \sum_{i=1}^n \left <v_i, v_j \right > x_i x_i
$$

$$
= {1 \over 2} \left ( \sum_{i=1}^n \sum_{j=1}^n \sum_{f=1}^k v_{i,f} v_{j,f} - \sum_{i=1}^n \sum_{f=1}^k v_{i,f}v_{i,f} x_i x_i \right )
$$

$$
= {1 \over 2} \left ( \sum_{f=1}^k \right ) \left ( \left (\sum_{i=1}^n v_{i,f} x_i \right ) \left (\sum_{j=1}^n v_{j,f} x_j \right ) - \sum_{i=1}^n v_{i,f}^2 x_i^2 \right )
$$

$$
{1 \over 2} \sum_{f=1}^k \left ( \left ( \sum_{i=1}^n v_{i,f} x_i \right )^2 - \sum_{i=1}^n v_{i,f}^2 x_i^2 \right )
$$

### Learning Factorization Machines

The gradient vector taken for each of the weights, is

$$
% <![CDATA[
{ \delta \over \delta \theta } \hat{y}(x) =
\begin{cases}
1 & \text{if } \theta \text{ is } w_0 \\
x_i & \text{if } \theta \text{ is } w_i \\
x_i \sum_{j=1}^{n} v_{j,f} \cdot x_j - v_{i,f} \cdot x_i^2 & \text{if } \theta \text{ is } \theta_{i,f}
\end{cases} %]]>
$$

### Factorization Models as Predictors

### Regression

$\hat{y}(x)$ can be used directly as the predictor and the optimization
criterion is the minimal least square error on $D$.

### Usage

The `train()` function in the [fm-regression.dml](https://github.com/apache/systemml/blob/master/scripts/staging/fm-regression.dml) script, takes in the input variable matrix and the corresponding target vector with some input kept for validation during training.

``` java
train = function(matrix[double] X, matrix[double] y, matrix[double] X_val, matrix[double] y_val)
    return (matrix[double] w0, matrix[double] W, matrix[double] V) {
  /*
   * Trains the FM model.
   *
   * Inputs:
   *  - X     : n examples with d features, of shape (n, d)
   *  - y     : Target matrix, of shape (n, 1)
   *  - X_val : Input validation data matrix, of shape (n, d)
   *  - y_val : Target validation matrix, of shape (n, 1)
   *
   * Outputs:
   *  - w0, W, V : updated model parameters.
   *
   * Network Architecture:
   *
   * X --> [model] --> out --> l2_loss::backward(out, y) --> dout
   *
   */

   ...
   # 7.Call adam::update for all parameters
   [w0,mw0,vw0] = adam::update(w0, dw0, lr, beta1, beta2, epsilon, t, mw0, vw0);
   [W, mW, vW]  = adam::update(W, dW, lr, beta1, beta2, epsilon, t, mW, vW );
   [V, mV, vV]  = adam::update(V, dV, lr, beta1, beta2, epsilon, t, mV, vV );

}
```

Once the `train` function returns the  weights for the `fm` model, these are passed to the `predict` function.

``` java
predict = function(matrix[double] X, matrix[double] w0, matrix[double] W, matrix[double] V)
    return (matrix[double] out) {
  /*
   * Computes the predictions for the given inputs.
   *
   * Inputs:
   *  - X : n examples with d features, of shape (n, d).
   *  - w0, W, V : trained model parameters.
   *
   * Outputs:
   *  - out : target vector, y.
   */

    out = fm::forward(X, w0, W, V);

}
```

### Binary Classification

The sign of $\hat{y}(x)$ is used & the parameters are optimized for the hinge
loss or logit loss.

### Usage

The `train` function in the [fm-binclass.dml](https://github.com/apache/systemml/blob/master/scripts/staging/fm-binclass.dml)
script, takes in the input variable matrix and the corresponding target vector
with some input kept for validation during training. This script also contain
`train()` and `predict()` function as in the case of regression.
