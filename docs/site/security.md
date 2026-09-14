---
layout: site
title: Apache SystemDS Security
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

(Adapted from [Apache Spark](https://github.com/apache/spark/blob/master/SECURITY.md) and [Apache Kafka](https://github.com/apache/kafka/blob/trunk/SECURITY.md))

## Things You Need To Know

- **Security is off by default.** Security features like authentication are not enabled by default. When exposing an Apache SystemDS deployment to the internet or an untrusted network, it is important to secure the application against potential threats.
- **Different levels of threat in different deployments.** Apache SystemDS supports multiple backends and each one supports different levels of security in different environments. None of the backends are secure by default. Be sure to evaluate your environment and take the appropriate measure to secure your Apache SystemDS deployment.
- **Apache SystemDS assumes a trusted operator.** Anyone with shell access to the machine on which Apache SystemDS is deployed can read and modify both data and code. Apache SystemDS does not protect deployments from their own adminstrators.
- **Apache SystemDS trusts its JVM and classpaths.** JARs included in the classpath of a process inherit full privileges of this process. Extensions of the classpath or loading supplementary components is equivalent to trusting these additions; the model assumes no hostile JARs are present.

## Local Backend

The local backend of Apache SystemDS enables executing operations on the local machine. Even though this does not require a network connection, there are still certain security risks that should be taken into account. Therefore, it is of high importance to explicitly ensure the security of the system before deploying Apache SystemDS.

## Apache Spark Backend

Apache SystemDS supports distributed execution using [Apache Spark 3.5.7](https://github.com/apache/spark/releases/tag/v3.5.7). Since Apache SystemDS incorporates Apache Spark, the [security model of Apache Spark](https://spark.apache.org/docs/3.5.7/security.html) as well as the security specifications of underlying dependencies also apply to Apache SystemDS.

## Federated Backend

The federated backend in Apache SystemDS allows to execute operations on data that is located on remote workers. This federated mode is intended for deployment within an enterprise network only, where participants can be reached over network only by trusted parties.

### SSL Authentication and Encryption

Apache SystemDS supports SSL authentication and encryption for the communication between the coordinators and federated workers in a federated deployment. Note that this feature is disabled by default. It can be enabled and configured through the following configuration options:

| Property Name | Default | Meaning | Since Version |
| ------------- | ------- | ------- | ------------- |
| sysds.federated.ssl | false | Boolean flag to disable and enable SSL encryption. | 3.5.0 |
| sysds.federated.ssl.cert | null | File path to the worker X.509 certificate chain. | 3.5.0 |
| sysds.federated.ssl.key | null | File path to the worker private PKCS key. | 3.5.0 |
| sysds.federated.ssl.trust | null | File path to the trusted certificates (CA). | 3.5.0 |

## Parameter Server with Homomorphic Encryption

Parameter servers in Apache SystemDS support the privacy-preserving technique of homomorphic encryption using the library [SEAL 3.7.0](https://github.com/microsoft/SEAL/releases/tag/v3.7.0). Note that this feature is disabled by default. When enabling homorphic encryption, the [security model of SEAL](https://github.com/microsoft/SEAL/blob/v3.7.0/SECURITY.md) applies also to Apache SystemDS.

## Reporting Security Issues

Suspected vulnerabilities should be sent privately to [security@apache.org](mailto:security%40apache.org). Please do not disclose the security issue publicly until the PMC has had time to investigate the issue, prepare a fix, and coordinate a release.
