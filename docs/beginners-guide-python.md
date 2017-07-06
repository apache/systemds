---
layout: global
title: Beginner's Guide for Python Users
description: Beginner's Guide for Python Users
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
To understand more about DML and PyDML, we recommend that you read [Beginner's Guide to DML and PyDML](https://apache.github.io/systemml/beginners-guide-to-dml-and-pydml.html).

For convenience of Python users, SystemML exposes several language-level APIs that allow Python users to use SystemML
and its algorithms without the need to know DML or PyDML. We explain these APIs in the below sections with example usecases.

## Download & Setup

Before you get started on SystemML, make sure that your environment is set up and ready to go.

### Install Java (need Java 8) and Apache Spark

If you already have an Apache Spark installation, you can skip this step.
  
<div class="codetabs">
<div data-lang="OSX" markdown="1">
```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew tap caskroom/cask
brew install Caskroom/cask/java
brew tap homebrew/versions
brew install apache-spark16
```
</div>
<div data-lang="Linux" markdown="1">
```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install)"
brew tap caskroom/cask
brew install Caskroom/cask/java
brew tap homebrew/versions
brew install apache-spark16
```
</div>
</div>

### Install SystemML

To install released SystemML, please use following commands:

<div class="codetabs">
<div data-lang="Python 2" markdown="1">
```bash
pip install systemml
```
</div>
<div data-lang="Python 3" markdown="1">
```bash
pip3 install systemml
```
</div>
</div>


If you want to try out the bleeding edge version, please use following commands: 

<div class="codetabs">
<div data-lang="Python 2" markdown="1">
```bash
git checkout https://github.com/apache/systemml.git
cd systemml
mvn clean package -P distribution
pip install target/systemml-1.0.0-SNAPSHOT-python.tgz
```
</div>
<div data-lang="Python 3" markdown="1">
```bash
git checkout https://github.com/apache/systemml.git
cd systemml
mvn clean package -P distribution
pip3 install target/systemml-1.0.0-SNAPSHOT-python.tgz
```
</div>
</div>

### Uninstall SystemML
To uninstall SystemML, please use following command:

<div class="codetabs">
<div data-lang="Python 2" markdown="1">
```bash
pip uninstall systemml
```
</div>
<div data-lang="Python 3" markdown="1">
```bash
pip3 uninstall systemml
```
</div>
</div>

### Start Pyspark shell

<div class="codetabs">
<div data-lang="Python 2" markdown="1">
```bash
pyspark
```
</div>
<div data-lang="Python 3" markdown="1">
```bash
PYSPARK_PYTHON=python3 pyspark
```
</div>
</div>

---

## Matrix operations

To get started with SystemML, let's try few elementary matrix multiplication operations:

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

```python
array([[-60.],
       [-60.],
       [-60.]])
```

Let us now write a simple script to train [linear regression](https://apache.github.io/systemml/algorithms-regression.html#linear-regression) 
model: $ \beta = solve(X^T X, X^T y) $. For simplicity, we will use direct-solve method and ignore
regularization parameter as well as intercept. 

```python
import numpy as np
from sklearn import datasets
import systemml as sml
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
X = sml.matrix(X_train)
y = sml.matrix(np.matrix(y_train).T)
A = X.transpose().dot(X)
b = X.transpose().dot(y)
beta = sml.solve(A, b).toNumPy()
y_predicted = X_test.dot(beta)
print('Residual sum of squares: %.2f' % np.mean((y_predicted - y_test) ** 2)) 
```

Output:

```bash
Residual sum of squares: 25282.12
```

We can improve the residual error by adding an intercept and regularization parameter. To do so, we
will use `mllearn` API described in the next section.

---

## Invoke SystemML's algorithms

SystemML also exposes a subpackage [mllearn](https://apache.github.io/systemml/python-reference#mllearn-api). This subpackage allows Python users to invoke SystemML algorithms
using Scikit-learn or MLPipeline API.  

### Scikit-learn interface

In the below example, we invoke SystemML's [Linear Regression](https://apache.github.io/systemml/algorithms-regression.html#linear-regression)
algorithm.
 
```python
import numpy as np
from sklearn import datasets
from systemml.mllearn import LinearRegression
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
regr = LinearRegression(spark, fit_intercept=True, C=float("inf"), solver='direct-solve')
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

Here is another example that where we invoke SystemML's [Logistic Regression](https://apache.github.io/systemml/algorithms-classification.html#multinomial-logistic-regression)
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

### Passing PySpark DataFrame

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

### MLPipeline interface

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

---

## Invoking DML/PyDML scripts using MLContext

The below example demonstrates how to invoke the algorithm [scripts/algorithms/MultiLogReg.dml](https://github.com/apache/systemml/blob/master/scripts/algorithms/MultiLogReg.dml)
using Python [MLContext API](https://apache.github.io/systemml/spark-mlcontext-programming-guide).

```python
from sklearn import datasets
from pyspark.sql import SQLContext
import systemml as sml
import pandas as pd
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target + 1
n_samples = len(X_digits)
# Split the data into training/testing sets and convert to PySpark DataFrame
X_df = sqlCtx.createDataFrame(pd.DataFrame(X_digits[:int(.9 * n_samples)]))
y_df = sqlCtx.createDataFrame(pd.DataFrame(y_digits[:int(.9 * n_samples)]))
ml = sml.MLContext(sc)
# Run the MultiLogReg.dml script at the given URL
scriptUrl = "https://raw.githubusercontent.com/apache/systemml/master/scripts/algorithms/MultiLogReg.dml"
script = sml.dml(scriptUrl).input(X=X_df, Y_vec=y_df).output("B_out")
beta = ml.execute(script).get('B_out').toNumPy()
```
