---
layout: base
title: SystemDS Documentation
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
  2. **Multiple execution modes**, including Spark MLContext, Spark Batch, Standalone, and JMLC.
  3. **Automatic optimization** based on data and cluster characteristics to ensure both efficiency and scalability.

This version of SystemDS supports: Java 17,  Python 3.5+, Hadoop 3.3.x, and Spark 3.5.x, Nvidia CUDA 10.2
 (CuDNN 7.x) Intel MKL (<=2019.x).

## Links

Various forms of documentation for SystemDS are available.

- a [DML Language Reference](./site/dml-language-reference) for an list of operations possible inside SystemDS.
- [Builtin Functions](./site/builtins-reference) contains a collection of builtin functions providing an high level abstraction on complex machine learning algorithms.
- [Algorithm Reference](./site/algorithms-reference) contains specifics on algorithms supported in systemds.
- [Entity Resolution](./site/entity-resolution) provides a collection of customizable entity resolution primitives and pipelines.
- [Run SystemDS](./site/run) contains an Helloworld example along with an environment setup guide.
- Instructions on python can be found at [Python Documentation](./api/python/index)
- The [JavaDOC](./api/java/index) contains internal documentation of the system source code.
- [Install from Source](./site/install) guides through setup from git download to running system.
- If you want to contribute take a look at [Contributing](https://github.com/apache/systemds/blob/main/CONTRIBUTING.md)
- [R to DML](./site/dml-vs-r-guide) walks through the basics of converting a script from R to dml.
