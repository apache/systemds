---
layout: global
title: Reference Guide for Python Users
description: Reference Guide for Python Users
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

SystemML enables flexible, scalable machine learning. This flexibility is achieved through the specification of a high-level declarative machine learning language that comes in two flavors, 
one with an R-like syntax (DML) and one with a Python-like syntax (PyDML).

Algorithm scripts written in DML and PyDML can be run on Hadoop, on Spark, or in Standalone mode. 
No script modifications are required to change between modes. SystemML automatically performs advanced optimizations 
based on data and cluster characteristics, so much of the need to manually tweak algorithms is largely reduced or eliminated.
To understand more about DML and PyDML, we recommend that you read [Beginner's Guide to DML and PyDML](https://apache.github.io/incubator-systemml/beginners-guide-to-dml-and-pydml.html).

For convenience of Python users, SystemML exposes several language-level APIs that allow Python users to use SystemML
and its algorithms without the need to know DML or PyDML. We explain these APIs in the below sections.

## matrix API

The matrix class allows users to perform linear algebra operations in SystemML using a NumPy-like interface.
This class supports several arithmetic operators (such as +, -, *, /, ^, etc).

matrix class is a python wrapper that implements basic matrix
operators, matrix functions as well as converters to common Python
types (for example: Numpy arrays, PySpark DataFrame and Pandas
DataFrame).

The operators supported are:

1.  Arithmetic operators: +, -, *, /, //, %, \** as well as dot
    (i.e. matrix multiplication)
2.  Indexing in the matrix
3.  Relational/Boolean operators: \<, \<=, \>, \>=, ==, !=, &, \|

In addition, following functions are supported for matrix:

1.  transpose
2.  Aggregation functions: sum, mean, var, sd, max, min, argmin,
    argmax, cumsum
3.  Global statistical built-In functions: exp, log, abs, sqrt,
    round, floor, ceil, sin, cos, tan, asin, acos, atan, sign, solve

For all the above functions, we always return a two dimensional matrix, especially for aggregation functions with axis. 
For example: Assuming m1 is a matrix of (3, n), NumPy returns a 1d vector of dimension (3,) for operation m1.sum(axis=1)
whereas SystemML returns a 2d matrix of dimension (3, 1).

Note: an evaluated matrix contains a data field computed by eval
method as DataFrame or NumPy array.

It is important to note that matrix class also supports most of NumPy's universal functions (i.e. ufuncs).
The current version of NumPy explicitly disables overriding ufunc, but this should be enabled in next release. 
Until then to test above code, please use:

```bash
git clone https://github.com/niketanpansare/numpy.git
cd numpy
python setup.py install
```

This will enable NumPy's functions to invoke matrix class:

```python
import systemml as sml
import numpy as np
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
np.add(m1, m2)
``` 

The matrix class doesnot support following ufuncs:

- Complex number related ufunc (for example: `conj`)
- Hyperbolic/inverse-hyperbolic functions (for example: sinh, arcsinh, cosh, ...)
- Bitwise operators
- Xor operator
- Infinite/Nan-checking (for example: isreal, iscomplex, isfinite, isinf, isnan)
- Other ufuncs: copysign, nextafter, modf, frexp, trunc.

This class also supports several input/output formats such as NumPy arrays, Pandas DataFrame, SciPy sparse matrix and PySpark DataFrame.

By default, the operations are evaluated lazily to avoid conversion overhead and also to maximize optimization scope.
To disable lazy evaluation, please us `set_lazy` method:

```python
>>> import systemml as sml
>>> import numpy as np
>>> m1 = sml.matrix(np.ones((3,3)) + 2)

Welcome to Apache SystemML!

>>> m2 = sml.matrix(np.ones((3,3)) + 3)
>>> np.add(m1, m2) + m1
# This matrix (mVar4) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPy() or toDF() or toPandas() methods.
mVar2 = load(" ", format="csv")
mVar1 = load(" ", format="csv")
mVar3 = mVar1 + mVar2
mVar4 = mVar3 + mVar1
save(mVar4, " ")


>>> sml.set_lazy(False)
>>> m1 = sml.matrix(np.ones((3,3)) + 2)
>>> m2 = sml.matrix(np.ones((3,3)) + 3)
>>> np.add(m1, m2) + m1
# This matrix (mVar8) is backed by NumPy array. To fetch the NumPy array, invoke toNumPy() method.
``` 

### Usage:

```python
import systemml as sml
import numpy as np
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
m2 = m1 * (m2 + m1)
m4 = 1.0 - m2
m4.sum(axis=1).toNumPy()
```

Output:

```bash
array([[-60.],
       [-60.],
       [-60.]])
```


### Design Decisions:

1.  Until eval() method is invoked, we create an AST (not exposed to
the user) that consist of unevaluated operations and data
required by those operations. As an anology, a spark user can
treat eval() method similar to calling RDD.persist() followed by
RDD.count().

2.  The AST consist of two kinds of nodes: either of type matrix or
of type DMLOp. Both these classes expose \_visit method, that
helps in traversing the AST in DFS manner.

3.  A matrix object can either be evaluated or not. If evaluated,
the attribute 'data' is set to one of the supported types (for
example: NumPy array or DataFrame). In this case, the attribute
'op' is set to None. If not evaluated, the attribute 'op' which
refers to one of the intermediate node of AST and if of type
DMLOp. In this case, the attribute 'data' is set to None.

5.  DMLOp has an attribute 'inputs' which contains list of matrix
objects or DMLOp.

6.  To simplify the traversal, every matrix object is considered
immutable and an matrix operations creates a new matrix object.
As an example: m1 = sml.matrix(np.ones((3,3))) creates a matrix
object backed by 'data=(np.ones((3,3))'. m1 = m1 \* 2 will
create a new matrix object which is now backed by 'op=DMLOp( ...)' 
whose input is earlier created matrix object.

7.  Left indexing (implemented in \_\_setitem\_\_ method) is a
special case, where Python expects the existing object to be
mutated. To ensure the above property, we make deep copy of
existing object and point any references to the left-indexed
matrix to the newly created object. Then the left-indexed matrix
is set to be backed by DMLOp consisting of following pydml:
left-indexed-matrix = new-deep-copied-matrix
left-indexed-matrix[index] = value

8.  Please use m.print\_ast() and/or type m for debugging. Here is a
sample session:

```python
>>> npm = np.ones((3,3))
>>> m1 = sml.matrix(npm + 3)
>>> m2 = sml.matrix(npm + 5)
>>> m3 = m1 + m2
>>> m3
mVar2 = load(" ", format="csv")
mVar1 = load(" ", format="csv")
mVar3 = mVar1 + mVar2
save(mVar3, " ")
>>> m3.print_ast()
- [mVar3] (op).
  - [mVar1] (data).
  - [mVar2] (data).    
```

## MLContext API

The Spark MLContext API offers a programmatic interface for interacting with SystemML from Spark using languages such as Scala, Java, and Python. 
As a result, it offers a convenient way to interact with SystemML from the Spark Shell and from Notebooks such as Jupyter and Zeppelin.

### Usage

The below example demonstrates how to invoke the algorithm [scripts/algorithms/MultiLogReg.dml](https://github.com/apache/incubator-systemml/blob/master/scripts/algorithms/MultiLogReg.dml)
using Python [MLContext API](https://apache.github.io/incubator-systemml/spark-mlcontext-programming-guide).

```python
from sklearn import datasets, neighbors
from pyspark.sql import DataFrame, SQLContext
import systemml as sml
import pandas as pd
import os, imp
sqlCtx = SQLContext(sc)
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target + 1
n_samples = len(X_digits)
# Split the data into training/testing sets and convert to PySpark DataFrame
X_df = sqlCtx.createDataFrame(pd.DataFrame(X_digits[:.9 * n_samples]))
y_df = sqlCtx.createDataFrame(pd.DataFrame(y_digits[:.9 * n_samples]))
ml = sml.MLContext(sc)
# Get the path of MultiLogReg.dml
scriptPath = os.path.join(imp.find_module("systemml")[1], 'systemml-java', 'scripts', 'algorithms', 'MultiLogReg.dml')
script = sml.dml(scriptPath).input(X=X_df, Y_vec=y_df).output("B_out")
beta = ml.execute(script).get('B_out').toNumPy()
```


## mllearn API

The below code describes how to use mllearn API for training:

<div class="codetabs">
<div data-lang="sklearn way" markdown="1">
{% highlight python %}
# Input: Two Python objects (X_train, y_train) of type numpy, pandas or scipy.
model.fit(X_train, y_train)
{% endhighlight %}
</div>
<div data-lang="mllib way" markdown="1">
{% highlight python %}
# Input: One LabeledPoint DataFrame with atleast two columns: features (of type Vector) and labels.
model.fit(X_df)
{% endhighlight %}
</div>
</div>

The below code describes how to use mllearn API for prediction:

<div class="codetabs">
<div data-lang="sklearn way" markdown="1">
{% highlight python %}
# Input: One Python object (X_test) of type numpy, pandas or scipy.
model.predict(X_test)
# OR model.score(X_test, y_test)
{% endhighlight %}
</div>
<div data-lang="mllib way" markdown="1">
{% highlight python %}
# Input: One LabeledPoint DataFrame (df_test) with atleast one column: features (of type Vector).
model.transform(df_test)
{% endhighlight %}
</div>
</div>


The table below describes the parameter available for mllearn algorithms:

| Parameters | Description of the Parameters | LogisticRegression | LinearRegression | SVM | NaiveBayes |
|----------------|-----------------------------------------------------------------------------------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|------------|
| sparkSession | PySpark SparkSession | X | X | X | X |
| penalty | Used to specify the norm used in the penalization (default: 'l2') | only 'l2' supported | - | - | - |
| fit_intercept | Specifies whether to add intercept or not (default: True) | X | X | X | - |
| normalize | This parameter is ignored when fit_intercept is set to False. (default: False) | X | X | X | - |
| max_iter | Maximum number of iterations (default: 100) | X | X | X | - |
| max_inner_iter | Maximum number of inner iterations, or 0 if no maximum limit provided (default: 0) | X | - | - | - |
| tol | Tolerance used in the convergence criterion (default: 0.000001) | X | X | X | - |
| C | 1/regularization parameter (default: 1.0). To disable regularization, please use float("inf") | X | X | X | - |
| solver | Algorithm to use in the optimization problem. | Only 'newton-cg' solver supported | Supports either 'newton-cg' or 'direct-solve' (default: 'newton-cg'). Depending on the size and the sparsity of the feature matrix, one or the other solver may be more efficient. 'direct-solve' solver is more efficient when the number of features is relatively small (m < 1000) and input matrix X is either tall or fairly dense; otherwise 'newton-cg' solver is more efficient. | - | - |
| is_multi_class | Specifies whether to use binary-class or multi-class classifier (default: False) | - | - | X | - |
| laplace | Laplace smoothing specified by the user to avoid creation of 0 probabilities (default: 1.0) | - | - | - | X |

In the below example, we invoke SystemML's [Logistic Regression](https://apache.github.io/incubator-systemml/algorithms-classification.html#multinomial-logistic-regression)
algorithm on digits datasets.

```python
# Scikit-learn way
from sklearn import datasets, neighbors
from systemml.mllearn import LogisticRegression
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
n_samples = len(X_digits)
X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]
logistic = LogisticRegression(spark)
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))
```

Output:

```bash
LogisticRegression score: 0.927778
```

You can also save the trained model and load it later for prediction:

```python
# Assuming logistic.fit(X_train, y_train) is already invoked
logistic.save('logistic_model')
new_logistic = LogisticRegression(spark)
new_logistic.load('logistic_model')
print('LogisticRegression score: %f' % new_logistic.score(X_test, y_test))
```

#### Passing PySpark DataFrame

To train the above algorithm on larger dataset, we can load the dataset into DataFrame and pass it to the `fit` method:

```python
from sklearn import datasets
from systemml.mllearn import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score
import systemml as sml
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
n_samples = len(X_digits)
# Split the data into training/testing sets and convert to PySpark DataFrame
df_train = sml.convertToLabeledDF(sqlCtx, X_digits[:int(.9 * n_samples)], y_digits[:int(.9 * n_samples)])
X_test = spark.createDataFrame(pd.DataFrame(X_digits[int(.9 * n_samples):]))
logistic = LogisticRegression(spark)
logistic.fit(df_train)
y_predicted = logistic.predict(X_test)
y_predicted = y_predicted.select('prediction').toPandas().as_matrix().flatten()
y_test = y_digits[int(.9 * n_samples):]
print('LogisticRegression score: %f' % accuracy_score(y_test, y_predicted))
```

Output:

```bash
LogisticRegression score: 0.922222
```

#### MLPipeline interface

In the below example, we demonstrate how the same `LogisticRegression` class can allow SystemML to fit seamlessly into 
large data pipelines.

```python
# MLPipeline way
from pyspark.ml import Pipeline
from systemml.mllearn import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 2.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 2.0),
    (4, "b spark who", 1.0),
    (5, "g d a y", 2.0),
    (6, "spark fly", 1.0),
    (7, "was mapreduce", 2.0),
    (8, "e spark program", 1.0),
    (9, "a e c l", 2.0),
    (10, "spark compile", 1.0),
    (11, "hadoop software", 2.0)
], ["id", "text", "label"])
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=20)
lr = LogisticRegression(sqlCtx)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
model = pipeline.fit(training)
test = spark.createDataFrame([
    (12, "spark i j k"),
    (13, "l m n"),
    (14, "mapreduce spark"),
    (15, "apache hadoop")], ["id", "text"])
prediction = model.transform(test)
prediction.show()
```

Output:

```bash
+-------+---+---------------+------------------+--------------------+--------------------+----------+
|__INDEX| id|           text|             words|            features|         probability|prediction|
+-------+---+---------------+------------------+--------------------+--------------------+----------+
|    1.0| 12|    spark i j k|  [spark, i, j, k]|(20,[5,6,7],[2.0,...|[0.99999999999975...|       1.0|
|    2.0| 13|          l m n|         [l, m, n]|(20,[8,9,10],[1.0...|[1.37552128844736...|       2.0|
|    3.0| 14|mapreduce spark|[mapreduce, spark]|(20,[5,10],[1.0,1...|[0.99860290938153...|       1.0|
|    4.0| 15|  apache hadoop|  [apache, hadoop]|(20,[9,14],[1.0,1...|[5.41688748236143...|       2.0|
+-------+---+---------------+------------------+--------------------+--------------------+----------+
```




## Troubleshooting Python APIs

#### Unable to load SystemML.jar into current pyspark session.

While using SystemML's Python package through pyspark or notebook (SparkContext is not previously created in the session), the
below method is not required. However, if the user wishes to use SystemML through spark-submit and has not previously invoked 

 `systemml.defmatrix.setSparkContext`(*sc*)
:   Before using the matrix, the user needs to invoke this function if SparkContext is not previously created in the session.

    sc: SparkContext
    :   SparkContext

Example:

```python
import systemml as sml
import numpy as np
sml.setSparkContext(sc)
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
m2 = m1 * (m2 + m1)
m4 = 1.0 - m2
m4.sum(axis=1).toNumPy()
```

If SystemML was not installed via pip, you may have to download SystemML.jar and provide it to pyspark via `--driver-class-path` and `--jars`. 

#### matrix API is running slow when set_lazy(False) or when eval() is called often.

This is a known issue. The matrix API is slow in this scenario due to slow Py4J conversion from Java MatrixObject or Java RDD to Python NumPy or DataFrame.
To resolve this for now, we recommend writing the matrix to FileSystemML and using `load` function.

#### maximum recursion depth exceeded

SystemML matrix is backed by lazy evaluation and uses a recursive Depth First Search (DFS).
Python can throw `RuntimeError: maximum recursion depth exceeded` when the recursion of DFS exceeds beyond the limit 
set by Python. There are two ways to address it:

1. Increase the limit in Python:
 
	```python
	import sys
	some_large_number = 2000
	sys.setrecursionlimit(some_large_number)
	```

2. Evaluate the intermeditate matrix to cut-off large recursion.