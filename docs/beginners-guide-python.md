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


## Download & Setup

Before you get started on SystemML, make sure that your environment is set up and ready to go.

### Install Java and Spark
  
<div class="codetabs">
<div data-lang="OSX" markdown="1">
{% highlight bash %}
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# Install Java (need Java 8)
brew tap caskroom/cask
brew install Caskroom/cask/java

# Install Spark
brew install apache-spark
{% endhighlight %}
</div>
<div data-lang="Linux" markdown="1">
{% highlight bash %}
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install)"

# Install Java (need Java 8)
brew tap caskroom/cask
brew install Caskroom/cask/java

# Install Spark
brew install apache-spark
{% endhighlight %}
</div>
</div>

### Install SystemML

The simplest way to install SystemML is `pip`.

<div class="codetabs">
<div data-lang="OSX" markdown="1">
{% highlight bash %}
pip install SystemML

# Get SYSTEMML_HOME and SPARK_CLASSPATH
SYSTEMML_HOME=`python -c 'import imp; import os; print os.path.join(imp.find_module("SystemML")[1], "SystemML-java")'`
SPARK_CLASSPATH=$SYSTEMML_HOME"/SystemML.jar"

# Append the above variables to ~/.bashrc
# If you are using zsh, append it to ~/.zshrc; if you are using ksh, append to ~/.profile and if you are using csh or tcsh, append to ~/.login
echo '' >> ~/.bashrc
echo 'export SYSTEMML_HOME='$SYSTEMML_HOME >> ~/.bashrc
echo 'export SPARK_CLASSPATH=$SPARK_CLASSPATH:'$SPARK_CLASSPATH >> ~/.bashrc

# Alternatively, you can provide '--driver-class-path $SYSTEMML_HOME"/SystemML.jar"' argument to pyspark.
{% endhighlight %}
</div>
<div data-lang="Linux" markdown="1">
pip install SystemML

# Get SYSTEMML_HOME and SPARK_CLASSPATH
SYSTEMML_HOME=`python -c 'import imp; import os; print os.path.join(imp.find_module("SystemML")[1], "SystemML-java")'`
SPARK_CLASSPATH=$SYSTEMML_HOME"/SystemML.jar"

# Append the above variables to ~/.bashrc
# If you are using zsh, append it to ~/.zshrc; if you are using ksh, append to ~/.profile and if you are using csh or tcsh, append to ~/.login 
echo '' >> ~/.bashrc
echo 'export SYSTEMML_HOME='$SYSTEMML_HOME >> ~/.bashrc
echo 'export SPARK_CLASSPATH=$SPARK_CLASSPATH:'$SPARK_CLASSPATH >> ~/.bashrc

# Alternatively, you can provide '--driver-class-path $SYSTEMML_HOME"/SystemML.jar"' argument to pyspark.
</div>
</div>

## Matrix operations

The simplest way to get started with SystemML is to try simple matrix operations:
 
{% highlight python %}
import SystemML as sml
import numpy as np
sml.setSparkContext(sc)
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
m2 = m1 * (m2 + m1)
m4 = 1.0 - m2
m4.sum().toNumPyArray()
{% endhighlight %}

## Invoke SystemML's algorithms
 
SystemML also exposes a subpackage `mllearn`. This subpackage allows Python users to invoke SystemML algorithms
using Scikit-learn or MLPipeline API.  
 
In the below example, we invoke SystemML's [Logistic Regression](https://apache.github.io/incubator-systemml/algorithms-classification.html#multinomial-logistic-regression)
algorithm on scikit-learn's datasets.

{% highlight python %}
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
{% endhighlight %}

In the below example, we demonstrate how the same class can be support DataFrame (as well as Spark's
MLPipelines), thus allowing it to fit seamlessly into large data pipelines.

{% highlight python %}
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
{% endhighlight %}

## Invoking DML/PyDML scripts using MLContext

TODO
