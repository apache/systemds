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

SystemML is now an **Apache Incubator** project! Please see the [**Apache SystemML (incubating)**](http://systemml.apache.org/)
website for more information.

SystemML is a flexible, scalable machine learning system.
SystemML's distinguishing characteristics are:

  1. **Algorithm customizability via R-like and Python-like languages**.
  2. **Multiple execution modes**, including Standalone, Spark Batch, Spark MLContext, Hadoop Batch, and JMLC.
  3. **Automatic optimization** based on data and cluster characteristics to ensure both efficiency and scalability.

The [SystemML GitHub README](https://github.com/apache/incubator-systemml) describes
building, testing, and running SystemML.

## Running SystemML

* **Standalone** - Standalone mode allows data scientists to rapidly prototype algorithms on a single
machine in R-like and Python-like declarative languages.
  * The [SystemML GitHub README](https://github.com/apache/incubator-systemml) describes
  a linear regression example in Standalone Mode.
  * The [Quick Start Guide](quick-start-guide.html) provides additional examples of algorithm execution
  in Standalone Mode.
* **Spark Batch** - Algorithms are automatically optimized to run across Spark clusters.
  * See **Invoking SystemML in Spark Batch Mode** **(Coming soon)**.
* **Spark MLContext** - Spark MLContext is a programmatic API for running SystemML from Spark via Scala or Java.
  * See the [Spark MLContext Programming Guide](spark-mlcontext-programming-guide.html) for
  [**Spark Shell (Scala)**](spark-mlcontext-programming-guide.html#spark-shell-example),
  [Java](spark-mlcontext-programming-guide.html#java-example), and
  [**Zeppelin Notebook**](spark-mlcontext-programming-guide.html#zeppelin-notebook-example---linear-regression-algorithm)
  examples.
* **Hadoop Batch** - Algorithms are automatically optimized when distributed across Hadoop clusters.
  * See [Invoking SystemML in Hadoop Batch Mode](hadoop-batch-mode.html) for detailed information.
* **JMLC** - Java Machine Learning Connector.

## Language Guides

* [DML Language Reference](dml-language-reference.html) -
DML is a high-level R-like declarative language for machine learning.
* **PyDML Language Reference** **(Coming Soon)** -
PyDML is a high-level Python-like declarative language for machine learning.
* [Beginner's Guide to DML and PyDML](beginners-guide-to-dml-and-pydml.html) -
An introduction to the basics of DML and PyDML.

## ML Algorithms

* [Algorithms Reference](algorithms-reference.html) - The Algorithms Reference describes the
machine learning algorithms included with SystemML in detail.

## Tools

* [Debugger Guide](debugger-guide.html) - SystemML supports DML script-level debugging through a
command-line interface.
