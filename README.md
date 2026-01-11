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
{% end comment %}
-->

# Apache SystemDS

**Overview:** Apache SystemDS is an open-source machine learning (ML) system for the end-to-end 
data science lifecycle from data preparation and cleaning, over efficient ML model training, 
to debugging and serving. ML algorithms or pipelines are specified in a high-level language 
with R-like syntax or related Python and Java APIs (with many builtin primitives), and the 
system automatically generates hybrid runtime plans of local, in-memory operations and distributed
operations on Apache Spark. Additional backends exist for GPUs and federated learning. 

Resource | Links
---------|------
**Quick Start** | [Install, Quick Start and Hello World](https://apache.github.io/systemds/site/install.html)
**Documentation:** | [SystemDS Documentation](https://apache.github.io/systemds/)
**Python Documentation** | [Python SystemDS Documentation](https://apache.github.io/systemds/api/python/index.html)
**Issue Tracker** | [Jira Dashboard](https://issues.apache.org/jira/secure/Dashboard.jspa?selectPageId=12335852)


**Status and Build:** SystemDS is renamed from SystemML which is an **Apache Top Level Project**.
To build from source visit [SystemDS Install from source](https://apache.github.io/systemds/site/install.html)

[![Build](https://github.com/apache/systemds/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/build.yml)
[![Documentation](https://github.com/apache/systemds/actions/workflows/documentation.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/documentation.yml)
[![LicenseCheck](https://github.com/apache/systemds/actions/workflows/license.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/license.yml)
[![Java Tests](https://github.com/apache/systemds/actions/workflows/javaTests.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/javaTests.yml)
[![Java Coverage](https://codecov.io/gh/apache/systemds/graph/badge.svg?token=4YfvX8s6Dz&flag=java)](https://codecov.io/gh/apache/systemds)
[![Python Test](https://github.com/apache/systemds/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/python.yml)
[![Python Coverage](https://codecov.io/gh/apache/systemds/graph/badge.svg?token=4YfvX8s6Dz&flag=python)](https://codecov.io/gh/apache/systemds)
[![Total PyPI downloads](https://static.pepy.tech/personalized-badge/systemds?units=abbreviation&period=total&left_color=grey&right_color=blue&left_text=Total%20PyPI%20Downloads)](https://pepy.tech/project/systemds)
[![Monthly PyPI downloads](https://static.pepy.tech/personalized-badge/systemds?units=abbreviation&left_color=grey&right_color=blue&left_text=Monthly%20PyPI%20Downloads)](https://pepy.tech/project/systemds)
