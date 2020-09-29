---
layout: site
title: Running SystemDS
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

If you want to execute from source code follow the [Install from source](install) guide first.

## Setting SYSTEMDS_ROOT environment variable

In order to run SystemDS it is highly recomended to setup SystemDS root on path.
This works both from your development directory containing source code and if
you download a release of SystemDS.

The following example works if you open an terminal at the root of the downloaded release,
or a cloned repository. (You can also change the `$(pwd)` with the full path to the folder.)

```bash
export SYSTEMDS_ROOT=$(pwd)
export PATH=$SYSTEMDS_ROOT/bin:$PATH
```

It can be beneficial to enter these into your `~/.profile` or `~/.bashrc` for linux,
(but remember to change `$(pwd` to the full folder path)
or your environment variables in windows to enable reuse between terminals and restarts.

```bash 
echo 'export SYSTEMDS_ROOT='$(pwd) >> ~/.bashrc
echo 'export PATH=$SYSTEMDS_ROOT/bin:$PATH' >> ~/.bashrc
```

## Hello, World! example

To quickly verify that the system is setup correctly.
You can run a simple hello world, using the launch script.

Open an terminal and go to an empty folder, then execute the following.

```bash
# Create a hello World script
echo 'print("Hello, World!")' > hello.dml
# Execute hello world Script
systemds hello.dml
# Remove the hello.dml
rm hello.dml
```

## Running a real first example

To see SystemDS in action a simple example using the `Univar-stats.dml`
script can be executed. This example is taken from the
[SystemML documentation](http://apache.github.io/systemml/standalone-guide).
The relevant commands to run this example with SystemDS will be listed here.
See their documentation for further details.  

### Example preparations

```bash
# download test data
wget -P data/ http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data

# generate a metadata file for the dataset
echo '{"rows": 306, "cols": 4, "format": "csv"}' > data/haberman.data.mtd

# generate type description for the data
echo '1,1,1,2' > data/types.csv
echo '{"rows": 1, "cols": 4, "format": "csv"}' > data/types.csv.mtd
```

### Executing the DML script

```shell script
bin/systemds Univar-Stats.dml -nvargs X=data/haberman.data TYPES=data/types.csv STATS=data/univarOut.mtx CONSOLE_OUTPUT=TRUE
```

## Using Intel MKL native instructions

To use the MKL acceleration download and install the latest supported MKL library (<=2019.5) from [1],
set the environment variables with the MKL-provided script `. /opt/intel/bin/compilervars.sh intel64` (note the dot and 
the default install location) and set the option `sysds.native.blas` in `SystemDS-config.xml` to mkl.

[1]: https://software.intel.com/mkl "Intel Math Kernel Library"