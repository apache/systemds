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

**Overview:** SystemDS is a versatile system for the end-to-end data science lifecycle from data integration, cleaning,
and feature engineering, over efficient, local and distributed ML model training, to deployment and serving. To this
end, we aim to provide a stack of declarative languages with R-like syntax for (1) the different tasks of the data-science
lifecycle, and (2) users with different expertise. These high-level scripts are compiled into hybrid execution plans of 
local, in-memory CPU and GPU operations, as well as distributed operations on Apache Spark. In contrast to existing 
systems - that either provide homogeneous tensors or 2D Datasets - and in order to serve the entire data science lifecycle,
the underlying data model are DataTensors, i.e., tensors (multi-dimensional arrays) whose first dimension may have a 
heterogeneous and nested schema.

**Quick Start** [Install, Quick Start and Hello World](https://apache.github.io/systemml/site/install.html)

**Documentation:** [SystemDS Documentation](https://apache.github.io/systemml/)

**Python Documentation** [Python SystemDS Documentation](https://apache.github.io/systemml/api/python/index.html)

**Status and Build:** SystemDS is still in pre-alpha status. The original code base was forked from 
[**Apache SystemML**](http://systemml.apache.org/) 1.2 in September 2018. We will continue to support linear algebra 
programs over matrices, while replacing the underlying data model and compiler, as well as substantially extending the 
supported functionalities. Until the first release, you can build your own snapshot via Apache Maven:
 `mvn clean package -P distribution`.
  
[![Build](https://github.com/apache/systemml/workflows/Build/badge.svg?branch=master&event=push)](https://github.com/apache/systemml/actions?query=workflow%3A%22Build%22+branch%3Amaster+event%3Apush)
[![Documentation](https://github.com/apache/systemml/workflows/Documentation/badge.svg?branch=master&event=push)](https://github.com/apache/systemml/actions?query=workflow%3ADocumentation+branch%3Amaster+event%3Apush)
[![Component Test](https://github.com/apache/systemml/workflows/Component%20Test/badge.svg?branch=master&event=push)](https://github.com/apache/systemml/actions?query=workflow%3A%22Component+Test%22+branch%3Amaster+event%3Apush)
[![Application Test](https://github.com/apache/systemml/workflows/Application%20Test/badge.svg?branch=master&event=push)](https://github.com/apache/systemml/actions?query=workflow%3A%22Application+Test%22+branch%3Amaster+event%3Apush)
[![Function Test](https://github.com/apache/systemml/workflows/Function%20Test/badge.svg?branch=master&event=push)](https://github.com/apache/systemml/actions?query=workflow%3A%22Function+Test%22+branch%3Amaster+event%3Apush)
[![Python Test](https://github.com/apache/systemml/workflows/Python%20Test/badge.svg?branch=master&event=push)](https://github.com/apache/systemml/actions?query=workflow%3A%22Python+Test%22+branch%3Amaster+event%3Apush)
[![Federated Python Test](https://github.com/apache/systemml/workflows/Federated%20Python%20Test/badge.svg?branch=master&event=push)](https://github.com/apache/systemml/actions?query=workflow%3A%22Federated+Python+Test%22+branch%3Amaster+event%3Apush)
