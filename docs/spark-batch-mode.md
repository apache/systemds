---
layout: global
title: Invoking SystemML in Spark Batch Mode
description: Invoking SystemML in Spark Batch Mode
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


# Overview

Given that a primary purpose of SystemML is to perform machine learning on large distributed data
sets, one of the most important ways to invoke SystemML is Spark Batch. Here, we will look at this
mode in more depth.

**NOTE:** For a programmatic API to run and interact with SystemML via Scala or Python, please see the
[Spark MLContext Programming Guide](spark-mlcontext-programming-guide).

---

# Spark Batch Mode Invocation Syntax

SystemML can be invoked in Hadoop Batch mode using the following syntax:

    spark-submit SystemML.jar [-? | -help | -f <filename>] (-config <config_filename>) ([-args | -nvargs] <args-list>)

The DML script to invoke is specified after the `-f` argument. Configuration settings can be passed to SystemML
using the optional `-config ` argument. DML scripts can optionally take named arguments (`-nvargs`) or positional
arguments (`-args`). Named arguments are preferred over positional arguments. Positional arguments are considered
to be deprecated. All the primary algorithm scripts included with SystemML use named arguments.


**Example #1: DML Invocation with Named Arguments**

    spark-submit SystemML.jar -f scripts/algorithms/Kmeans.dml -nvargs X=X.mtx k=5


**Example #2: DML Invocation with Positional Arguments**

	spark-submit SystemML.jar -f src/test/scripts/applications/linear_regression/LinearRegression.dml -args "v" "y" 0.00000001 "w"

# Execution modes

SystemML works seamlessly with all Spark execution modes, including *local* (`--master local[*]`),
*yarn client* (`--master yarn-client`), *yarn cluster* (`--master yarn-cluster`), *etc*.  More
information on Spark cluster execution modes can be found on the
[official Spark cluster deployment documentation](https://spark.apache.org/docs/latest/cluster-overview.html).
*Note* that Spark can be easily run on a laptop in local mode using the `--master local[*]` described
above, which SystemML supports.

# Recommended Spark Configuration Settings

For best performance, we recommend setting the following flags when running SystemML with Spark:
`--conf spark.driver.maxResultSize=0 --conf spark.akka.frameSize=128`.

# Examples

Please see the MNIST examples in the included
[SystemML-NN](https://github.com/apache/systemml/tree/master/scripts/nn)
library for examples of Spark Batch mode execution with SystemML to train MNIST classifiers:

  * [MNIST Softmax Classifier](https://github.com/apache/systemml/blob/master/scripts/nn/examples/mnist_softmax-train.dml)
  * [MNIST LeNet ConvNet](https://github.com/apache/systemml/blob/master/scripts/nn/examples/mnist_lenet-train.dml)
