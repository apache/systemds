---
layout: global
displayTitle: SystemDS Documentation
title: SystemDS Documentation
description: SystemDS Documentation
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

SystemDS is a flexible, scalable machine learning system.
SystemDS's distinguishing characteristics are:

  1. **Algorithm customizability via R-like and Python-like languages**.
  2. **Multiple execution modes**, including Spark DSContext, Spark Batch, Hadoop Batch, Standalone, and JMLC.
  3. **Automatic optimization** based on data and cluster characteristics to ensure both efficiency and scalability.

The [SystemDS GitHub README](https://github.com/apache/systemml) describes
building, testing, and running SystemDS. Please read [Contributing to SystemDS](contributing-to-systemds)
to find out how to help make SystemDS even better!

To download SystemDS, visit the [downloads](http://systemml.apache.org/download) page.

This version of SystemDS supports: Java 8+, Scala 2.11+, Python 2.7/3.5+, Hadoop 2.6+, and Spark 2.1+.

## Quick tour of the documentation

* If you are new to SystemDS, please refer to the [installation guide](http://systemml.apache.org/install-systemml.html) and try out our [sample notebooks](http://systemml.apache.org/get-started.html#sample-notebook)
* If you want to invoke one of our [pre-implemented algorithms](algorithms-reference):
  * In Python, consider using 
    * the convenient [mllearn API](http://apache.github.io/systemml/python-reference.html#mllearn-api). The usage is described in our [beginner's guide](http://apache.github.io/systemml/beginners-guide-python.html#invoke-systemmls-algorithms)  
    * Or [Spark DSContext](spark-mlcontext-programming-guide) API
  * In Java/Scala, consider using 
    * [Spark DSContext](spark-mlcontext-programming-guide) API for large datasets
    * Or [JMLC](jmlc) API for in-memory scoring
  * Via Command-line, follow the usage section in the [Algorithms Reference](algorithms-reference) 
* If you want to implement a deep neural network, consider
  * Specifying your network in [Keras](https://keras.io/) format and invoking it with [Keras2DML](beginners-guide-keras2dml) API
  * Or specifying your network in [Caffe](http://caffe.berkeleyvision.org/) format and invoking it with [Caffe2DML](beginners-guide-caffe2dml) API
  * Or using DML-bodied [NN library](https://github.com/apache/systemml/tree/master/scripts/nn). The usage is described in our [sample notebook](https://github.com/apache/systemml/blob/master/samples/jupyter-notebooks/Deep%20Learning%20Image%20Classification.ipynb)
* Since training a deep neural network is often compute-bound, you may want to enable SystemDS's
  * [native BLAS](native-backend)
  * Or [GPU backend](gpu)
* If you want to implement a custom machine learning algorithm and you are familiar with:
  * R syntax, consider implementing your algorithm in [DML](dml-language-reference) (recommended)
  * Python syntax, you can implement your algorithm in [PyDML](beginners-guide-to-dml-and-pydml) or using the [matrix class](http://apache.github.io/systemml/python-reference.html#matrix-class)
* If you want to try out SystemDS on your laptop, consider
  * using the above mentioned APIs with Apache Spark (recommended). Please refer to our [installation guide](http://systemml.apache.org/install-systemml.html) for instructions on how to setup SystemDS on your laptop
  * Or running SystemDS in the [standalone mode](standalone-guide) with Java

## Running SystemDS

* [Beginner's Guide For Python Users](beginners-guide-python) - Beginner's Guide for Python users.
* [Spark DSContext](spark-mlcontext-programming-guide) - Spark DSContext is a programmatic API
for running SystemDS from Spark via Scala, Python, or Java.
  * [Spark Shell Example (Scala)](spark-mlcontext-programming-guide#spark-shell-example)
  * [Jupyter Notebook Example (PySpark)](spark-mlcontext-programming-guide#jupyter-pyspark-notebook-example---poisson-nonnegative-matrix-factorization)
* [Spark Batch](spark-batch-mode) - Algorithms are automatically optimized to run across Spark clusters.
* [Hadoop Batch](hadoop-batch-mode) - Algorithms are automatically optimized when distributed across Hadoop clusters.
* [Standalone](standalone-guide) - Standalone mode allows data scientists to rapidly prototype algorithms on a single
machine in R-like and Python-like declarative languages.
* [JMLC](jmlc) - Java Machine Learning Connector.
* [Deep Learning with SystemDS](deep-learning)
  * Keras2DML API for Deep Learning ([beginner's guide](beginners-guide-keras2dml), [reference guide](reference-guide-keras2dml)) - Converts a Keras model to DML.
  * Caffe2DML API for Deep Learning ([beginner's guide](beginners-guide-caffe2dml), [reference guide](reference-guide-caffe2dml)) - Converts a Caffe specification to DML.

## Language Guides

* [Python API Reference](python-reference) - API Reference Guide for Python users.
* [DML Language Reference](dml-language-reference) -
DML is a high-level R-like declarative language for machine learning.
* **PyDML Language Reference** -
PyDML is a high-level Python-like declarative language for machine learning.
* [Beginner's Guide to DML and PyDML](beginners-guide-to-dml-and-pydml) -
An introduction to the basics of DML and PyDML.

## ML Algorithms

* [Algorithms Reference](algorithms-reference) - The Algorithms Reference describes the
machine learning algorithms included with SystemDS in detail.

## Tools

* [Debugger Guide](debugger-guide) - SystemDS supports DML script-level debugging through a
command-line interface.
* [IDE Guide](developer-tools-systemds) - Useful IDE Guide for Developing SystemDS.

## Other

* [Contributing to SystemDS](contributing-to-systemds) - Describes ways to contribute to SystemDS.
* [Engine Developer Guide](engine-dev-guide) - Guide for internal SystemDS engine development.
* [Troubleshooting Guide](troubleshooting-guide) - Troubleshoot various issues related to SystemDS.
* [Release Process](release-process) - Description of the SystemDS release process.
* [Using Native BLAS](native-backend) in SystemDS.
* [Using GPU backend](gpu) in SystemDS.
