---
layout: site
title: SystemDS Quickstart Guide
description: Quickstart guide for installing and running SystemDS on Windows, Linux, and macOS
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

Welcome to the quickstart guide for Apache SystemDS. This quickstart page provides a high-level overview of both installation and points you to the detailed documentation for each path.

SystemDS can be installed and used in two different ways:

1. Using a **downloaded release**  
2. Using a **source build**

If you are primarily a user of SystemDS, start with the Release installation. If you plan to contribute or modify internals, follow the Source installation.

Each method is demonstrated in:
- Local mode  
- Spark mode  
- Federated mode (simple example)

For detailed configuration topics (BLAS, GPU, federated setup, contributing), see the links at the end.

---

# 1. Install from a Release

If you simply want to *use* SystemDS without modifying the source code, the recommended approach is to install SystemDS from an official Apache release.

**Full Release Installation Guide:** [Install SystemDS from a Release](release_install)

# 2. Install from Source

If you plan to contribute to SystemDS or need to modify its internals, you can build SystemDS from source.

**Full Source Build Guide:** [Install SystemDS from Source](source_install)

# 3. After Installation

Once either installation path is completed, you can start running scripts:

- Local Mode - Run SystemDS locally
- Spark Mode - Execute scripts on Spark through `spark-submit`
- Federated Mode - Run operations on remote data using federated workers

For detailed commands and examples: [Execute SystemDS](run)

# 4. More Configuration

SystemDS provides advanced configuration options for performance tuning and specialized execution environments. 

- GPU Support — [GPU Guide](https://apache.github.io/systemds/site/gpu)  
- BLAS / Native Acceleration — [Native Backend (BLAS) Guide](native-backend)  
- Federated Backend Deployment — [Federated Guide](federated-monitoring)  
- Contributing to SystemDS — [Contributing Guide](https://github.com/apache/systemds/blob/main/CONTRIBUTING.md)

