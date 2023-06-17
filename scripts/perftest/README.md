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

# Performance Tests SystemDS

To run all performance tests for SystemDS:
 * install systemds,
 * install the prerequisites,
 * navigate to the perftest directory $`cd $SYSTEMDS_ROOT/scripts/perftest` 
 * generate the data,
 * and execute.

There are a few prerequisites:

## Install SystemDS

- First follow the install guide: <http://apache.github.io/systemds/site/install> and build the project.
- Install the python package for python api benchmarks: <https://apache.github.io/systemds/api/python/getting_started/install.html>
- Prepare to run SystemDS: <https://apache.github.io/systemds/site/run>

## Install Additional Prerequisites
- Setup Intel MKL: <http://apache.github.io/systemds/site/run>
- Setup OpenBlas: <https://github.com/xianyi/OpenBLAS/wiki/Precompiled-installation-packages>
- Install Perf stat: <https://linoxide.com/linux-how-to/install-perf-tool-centos-ubuntu/>

## Generate Test Data

Using the scripts found in `$SYSTEMDS_ROOT/scripts/perftest/datagen`, generate the data for the tests you want to run. Note the sometimes optional and other times required parameters/args. Dataset size is likely the most important of these.

## Run the Benchmarks

**Reminder: The scripts should be run from the perftest folder.**

Examples:

```bash
./runAll.sh
```

Or look inside the runAll script to see how to run individual tests.

Time calculations in the bash scripts may additionally subtract a number, e.g. ".4".
This is done to accommodate for time lost by shell script and JVM startup overheads, to match the actual application runtime of SystemML.
