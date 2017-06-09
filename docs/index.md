---
layout: global
displayTitle: SystemML Documentation
title: SystemML Documentation
description: SystemML Documentation
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

SystemML is a flexible, scalable machine learning system.
SystemML's distinguishing characteristics are:

  1. **Algorithm customizability via R-like and Python-like languages**.
  2. **Multiple execution modes**, including Spark MLContext, Spark Batch, Hadoop Batch, Standalone, and JMLC.
  3. **Automatic optimization** based on data and cluster characteristics to ensure both efficiency and scalability.

The [SystemML GitHub README](https://github.com/apache/systemml) describes
building, testing, and running SystemML. Please read [Contributing to SystemML](contributing-to-systemml)
to find out how to help make SystemML even better!

To download SystemML, visit the [downloads](http://systemml.apache.org/download) page.

This version of SystemML supports: Java 8+, Scala 2.11+, Python 2.7/3.5+, Hadoop 2.6+, and Spark 2.1+.

## Running SystemML

* [Beginner's Guide For Python Users](beginners-guide-python) - Beginner's Guide for Python users.
* [Spark MLContext](spark-mlcontext-programming-guide) - Spark MLContext is a programmatic API
for running SystemML from Spark via Scala, Python, or Java.
  * [Spark Shell Example (Scala)](spark-mlcontext-programming-guide#spark-shell-example)
  * [Jupyter Notebook Example (PySpark)](spark-mlcontext-programming-guide#jupyter-pyspark-notebook-example---poisson-nonnegative-matrix-factorization)
* [Spark Batch](spark-batch-mode) - Algorithms are automatically optimized to run across Spark clusters.
* [Hadoop Batch](hadoop-batch-mode) - Algorithms are automatically optimized when distributed across Hadoop clusters.
* [Standalone](standalone-guide) - Standalone mode allows data scientists to rapidly prototype algorithms on a single
machine in R-like and Python-like declarative languages.
* [JMLC](jmlc) - Java Machine Learning Connector.
* *Experimental* [Caffe2DML API](beginners-guide-caffe2dml) for Deep Learning.

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
machine learning algorithms included with SystemML in detail.

## Tools

* [Debugger Guide](debugger-guide) - SystemML supports DML script-level debugging through a
command-line interface.
* [IDE Guide](developer-tools-systemml) - Useful IDE Guide for Developing SystemML.

## Other

* [Contributing to SystemML](contributing-to-systemml) - Describes ways to contribute to SystemML.
* [Engine Developer Guide](engine-dev-guide) - Guide for internal SystemML engine development.
* [Troubleshooting Guide](troubleshooting-guide) - Troubleshoot various issues related to SystemML.
* [Release Process](release-process) - Description of the SystemML release process.
