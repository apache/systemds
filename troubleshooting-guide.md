---
layout: global
title: Troubleshooting Guide
description: Troubleshooting Guide
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


## ClassNotFoundException for commons-math3

The Apache Commons Math library is utilized by SystemML. The commons-math3
dependency is included with Spark and with newer versions of Hadoop. Running
SystemML on an older Hadoop cluster can potentially generate an error such
as the following due to the missing commons-math3 dependency:

	java.lang.ClassNotFoundException: org.apache.commons.math3.linear.RealMatrix

This issue can be fixed by changing the commons-math3 `scope` in the pom.xml file
from `provided` to `compile`.

	<dependency>
		<groupId>org.apache.commons</groupId>
		<artifactId>commons-math3</artifactId>
		<version>3.1.1</version>
		<scope>compile</scope>
	</dependency>

SystemML can then be rebuilt with the `commons-math3` dependency using
Maven (`mvn clean package -P distribution`).

