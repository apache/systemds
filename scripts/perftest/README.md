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

# Perf tests SystemDS

to run all performance tests for SystemDS, simply download systemds, install the prerequisites and execute.

There are a few prerequisites:

- First follow the install guide: <http://apache.github.io/systemds/site/install>
- Setup Intel MKL: <http://apache.github.io/systemds/site/run>
- Install Perf stat: <https://linoxide.com/linux-how-to/install-perf-tool-centos-ubuntu/>

```bash
./scripts/perftest/runAll.sh
```

look inside the runAll script to see how to run individual tests.
