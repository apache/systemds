---
layout: global
title: Beginner's Guide for Python users
description: Beginner's Guide for Python users
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
and its algorithms without the need to know DML or PyDML. We explain these APIs in the below sections with example usecases.

## Download & Setup

Before you get started on SystemML, make sure that your environment is set up and ready to go.

### Install Java (need Java 8) and Apache Spark

If you already have a Apache Spark installation, you can skip this step.
  
<div class="codetabs">
<div data-lang="OSX" markdown="1">
```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew tap caskroom/cask
brew install Caskroom/cask/java
brew install apache-spark
```
</div>
<div data-lang="Linux" markdown="1">
```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install)"
brew tap caskroom/cask
brew install Caskroom/cask/java
brew install apache-spark
```
</div>
</div>

### Install SystemML

#### Step 1: Install SystemML Python package 

```bash
pip install SystemML
```

#### Step 2: Download SystemML Java binaries

SystemML Python package downloads the corresponding Java binaries (along with algorithms) and places them 
into the installed location. To find the location of the downloaded Java binaries, use the following command:

```bash
python -c 'import imp; import os; print os.path.join(imp.find_module("SystemML")[1], "SystemML-java")'
```

#### Step 3: (Optional but recommended) Set SYSTEMML_HOME environment variable
<div class="codetabs">
<div data-lang="OSX" markdown="1">
```bash
SYSTEMML_HOME=`python -c 'import imp; import os; print os.path.join(imp.find_module("SystemML")[1], "SystemML-java")'`
# If you are using zsh or ksh or csh, append it to ~/.zshrc or ~/.profile or ~/.login respectively.
echo '' >> ~/.bashrc
echo 'export SYSTEMML_HOME='$SYSTEMML_HOME >> ~/.bashrc
```
</div>
<div data-lang="Linux" markdown="1">
```bash
SYSTEMML_HOME=`python -c 'import imp; import os; print os.path.join(imp.find_module("SystemML")[1], "SystemML-java")'`
# If you are using zsh or ksh or csh, append it to ~/.zshrc or ~/.profile or ~/.login respectively.
echo '' >> ~/.bashrc
echo 'export SYSTEMML_HOME='$SYSTEMML_HOME >> ~/.bashrc
```
</div>
</div>

Note: the user is free to either use the prepackaged Java binaries 
or download them from [SystemML website](http://systemml.apache.org/download.html) 
or build them from the [source](https://github.com/apache/incubator-systemml).

### Start Pyspark shell

<div class="codetabs">
<div data-lang="OSX" markdown="1">
```bash
pyspark --master local[*] --driver-class-path $SYSTEMML_HOME"/SystemML.jar"
```
</div>
<div data-lang="Linux" markdown="1">
```bash
pyspark --master local[*] --driver-class-path $SYSTEMML_HOME"/SystemML.jar"
```
</div>
</div>

## Matrix operations

To get started with SystemML, let's try few elementary matrix multiplication operations:

```python
import SystemML as sml
import numpy as np
sml.setSparkContext(sc)
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
m2 = m1 * (m2 + m1)
m4 = 1.0 - m2
m4.sum(axis=1).toNumPyArray()
```

Output:

```bash
array([[-60.],
       [-60.],
       [-60.]])
```

Let us now write a simple script to train [linear regression](https://apache.github.io/incubator-systemml/algorithms-regression.html#linear-regression) 
model: $ \beta = solve(X^T X, X^T y) $. For simplicity, we will use direct-solve method and ignore regularization parameter as well as intercept. 

```python
import numpy as np
from sklearn import datasets
import SystemML as sml
from pyspark.sql import SQLContext
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
# Split the data into training/testing sets
X_train = diabetes_X[:-20]
X_test = diabetes_X[-20:]
# Split the targets into training/testing sets
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]
# Train Linear Regression model
sml.setSparkContext(sc)
X = sml.matrix(X_train)
y = sml.matrix(y_train)
A = X.transpose().dot(X)
b = X.transpose().dot(y)
beta = sml.solve(A, b).toNumPyArray()
y_predicted = X_test.dot(beta)
print('Residual sum of squares: %.2f' % np.mean((y_predicted - y_test) ** 2)) 
```

Output:

```bash
Residual sum of squares: 25282.12
```

We can improve the residual error by adding an intercept and regularization parameter. To do so, we will use `mllearn` API described in the next section.

## Invoke SystemML's algorithms

SystemML also exposes a subpackage `mllearn`. This subpackage allows Python users to invoke SystemML algorithms
using Scikit-learn or MLPipeline API.  

### Scikit-learn interface

In the below example, we invoke SystemML's [Linear Regression](https://apache.github.io/incubator-systemml/algorithms-regression.html#linear-regression)
algorithm.
 
```python
import numpy as np
from sklearn import datasets
from SystemML.mllearn import LinearRegression
from pyspark.sql import SQLContext
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
# Split the data into training/testing sets
X_train = diabetes_X[:-20]
X_test = diabetes_X[-20:]
# Split the targets into training/testing sets
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]
# Create linear regression object
regr = LinearRegression(sqlCtx, fit_intercept=True, C=1, solver='direct-solve')
# Train the model using the training sets
regr.fit(X_train, y_train)
y_predicted = regr.predict(X_test)
print('Residual sum of squares: %.2f' % np.mean((y_predicted - y_test) ** 2)) 
```

Output:

```bash
Residual sum of squares: 6991.17
```

As expected, by adding intercept and regularizer the residual error drops significantly.

Here is another example that where we invoke SystemML's [Logistic Regression](https://apache.github.io/incubator-systemml/algorithms-classification.html#multinomial-logistic-regression)
algorithm on digits datasets.

```python
# Scikit-learn way
from sklearn import datasets, neighbors
from SystemML.mllearn import LogisticRegression
from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target + 1
n_samples = len(X_digits)
X_train = X_digits[:.9 * n_samples]
y_train = y_digits[:.9 * n_samples]
X_test = X_digits[.9 * n_samples:]
y_test = y_digits[.9 * n_samples:]
logistic = LogisticRegression(sqlCtx)
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))
```

### Passing PySpark DataFrame

To train the above algorithm on larger dataset, we can load the dataset into DataFrame and pass it to the `fit` method:

```python
from sklearn import datasets, neighbors
from SystemML.mllearn import LogisticRegression
from pyspark.sql import SQLContext
import SystemML as sml
sqlCtx = SQLContext(sc)
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target + 1
n_samples = len(X_digits)
# Split the data into training/testing sets and convert to PySpark DataFrame
df_train = sml.convertToLabeledDF(sqlContext, X_digits[:.9 * n_samples], y_digits[:.9 * n_samples])
X_test = X_digits[.9 * n_samples:]
y_test = y_digits[.9 * n_samples:]
logistic = LogisticRegression(sqlCtx)
print('LogisticRegression score: %f' % logistic.fit(df_train).score(X_test, y_test))
```

### MLPipeline interface

In the below example, we demonstrate how the same `LogisticRegression` class can allow SystemML to fit seamlessly into 
large data pipelines.

```python
# MLPipeline way
from pyspark.ml import Pipeline
from SystemML.mllearn import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
training = sqlCtx.createDataFrame([
    (0L, "a b c d e spark", 1.0),
    (1L, "b d", 2.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 2.0),
    (4L, "b spark who", 1.0),
    (5L, "g d a y", 2.0),
    (6L, "spark fly", 1.0),
    (7L, "was mapreduce", 2.0),
    (8L, "e spark program", 1.0),
    (9L, "a e c l", 2.0),
    (10L, "spark compile", 1.0),
    (11L, "hadoop software", 2.0)
], ["id", "text", "label"])
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=20)
lr = LogisticRegression(sqlCtx)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
model = pipeline.fit(training)
test = sqlCtx.createDataFrame([
    (12L, "spark i j k"),
    (13L, "l m n"),
    (14L, "mapreduce spark"),
    (15L, "apache hadoop")], ["id", "text"])
prediction = model.transform(test)
prediction.show()
```

## Invoking DML/PyDML scripts using MLContext

The below example demonstrates how to invoke the algorithm [scripts/algorithms/MultiLogReg.dml](https://github.com/apache/incubator-systemml/blob/master/scripts/algorithms/MultiLogReg.dml)
using Python [MLContext API](https://apache.github.io/incubator-systemml/spark-mlcontext-programming-guide).

```python
from sklearn import datasets, neighbors
from pyspark.sql import DataFrame, SQLContext
import SystemML as sml
import pandas as pd
import os
sqlCtx = SQLContext(sc)
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target + 1
n_samples = len(X_digits)
# Split the data into training/testing sets and convert to PySpark DataFrame
X_df = sqlCtx.createDataFrame(pd.DataFrame(X_digits[:.9 * n_samples]))
y_df = sqlCtx.createDataFrame(pd.DataFrame(y_digits[:.9 * n_samples]))
ml = sml.MLContext(sc)
script = os.path.join(os.environ['SYSTEMML_HOME'], 'scripts', 'algorithms', 'MultiLogReg.dml')
script = sml.dml(script).input(X=X_df, Y_vec=y_df).input(**{"$X": ' ', "$Y": ' ', "$B": ' '}).out("B_out")
beta = ml.execute(script).getNumPyArray('B_out')
```
