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

**Overview:** SystemDS is an open source ML system for the end-to-end data science lifecycle from data integration, cleaning,
and feature engineering, over efficient, local and distributed ML model training, to deployment and serving. To this
end, we aim to provide a stack of declarative languages with R-like syntax for (1) the different tasks of the data-science
lifecycle, and (2) users with different expertise. These high-level scripts are compiled into hybrid execution plans of
local, in-memory CPU and GPU operations, as well as distributed operations on Apache Spark. In contrast to existing
systems - that either provide homogeneous tensors or 2D Datasets - and in order to serve the entire data science lifecycle,
the underlying data model are DataTensors, i.e., tensors (multi-dimensional arrays) whose first dimension may have a
heterogeneous and nested schema.


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
[![codecov](https://codecov.io/gh/apache/systemds/graph/badge.svg?token=4YfvX8s6Dz)](https://codecov.io/gh/apache/systemds)
[![Python Test](https://github.com/apache/systemds/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/apache/systemds/actions/workflows/python.yml)
[![Total PyPI downloads](https://static.pepy.tech/personalized-badge/systemds?units=abbreviation&period=total&left_color=grey&right_color=blue&left_text=Total%20PyPI%20Downloads)](https://pepy.tech/project/systemds)
[![Monthly PyPI downloads](https://static.pepy.tech/personalized-badge/systemds?units=abbreviation&left_color=grey&right_color=blue&left_text=Monthly%20PyPI%20Downloads)](https://pepy.tech/project/systemds)
