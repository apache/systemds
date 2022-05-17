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

# Performance tests SystemDS

To run all performance tests for SystemDS, simply download systemds, install the prerequisites and execute.

There are a few prerequisites:

- First follow the install guide: <http://apache.github.io/systemds/site/install> and build the project.
- Setup Intel MKL: <http://apache.github.io/systemds/site/run>
- Setup OpenBlas: <https://github.com/xianyi/OpenBLAS/wiki/Precompiled-installation-packages>
- Install Perf stat: <https://linoxide.com/linux-how-to/install-perf-tool-centos-ubuntu/>

## NOTE THE SCRIPT HAS TO BE RUN FROM THE PERFTEST FOLDER

Examples:

```bash
./runAll.sh
```

Look inside the runAll script to see how to run individual tests.

Time calculations in the bash scripts additionally subtract a number, e.g. ".4".
This is done to accommodate for time lost by shell script and JVM startup overheads, to match the actual application runtime of SystemML.
