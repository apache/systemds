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

# SystemDS

## Overview

SystemDS is a versatile system for the end-to-end data science lifecycle from data integration, cleaning, and feature engineering, over efficient, local and distributed ML model training, to deployment and serving. To this end, we aim to provide a stack of declarative languages with R-like syntax for (1) the different tasks of the data-science lifecycle, and (2) users with different expertise. These high-level scripts are compiled into hybrid execution plans of local, in-memory CPU and GPU operations, as well as distributed operations on Apache Spark. In contrast to existing systems - that either provide homogeneous tensors or 2D Datasets - and in order to serve the entire data science lifecycle, the underlying data model are DataTensors, i.e., tensors (multi-dimensional arrays) whose first dimension may have a heterogeneous and nested schema.

**Documentation:** [SystemDS Documentation](https://github.com/apache/systemds/tree/main/docs)

## Getting started

Requirements for running SystemDS are a bash shell and OpenJDK 17 or a Spark 3.5.x cluster installation (to run distributed jobs).
These requirements should be available via standard system packages in all major Linux distributions
(make sure to have the right JDK version enabled, if you have multiple versions in your system).
For Windows, a bash comes with [git for windows](http://git-scm.com) and OpenJDK builds can be obtained at <http://adoptopenjdk.net>
(tested version [jdk8u232-b09](https://adoptopenjdk.net/archive.html))  

To start out with an example after having installed the requirements mentioned above, create a text file  
`hello.dml` in your unzipped SystemDS directory containing the following content:

```shell script
X = rand(rows=$1, cols=$2, min=0, max=10, sparsity=$3)
Y = rand(rows=$2, cols=$1, min=0, max=10, sparsity=$3)
Z = X %*% Y
print("Your hello world matrix contains:")
print(toString(Z))
write(Z, "Z")
```

**Explanation:** The script takes three parameters for the creation of your matrices X and Y: rows, columns and degree
of sparsity. As you can see, DML can access these parameters by specifying $1, $2, ... etc

**Execution:** Now run that first script you created by running one of the following commands depending on your operating system:

### Running a script locally

``` bash
./bin/systemds hello.dml -args 10 10 1.0
```

### Running a script locally, providing your own SystemDS.jar file

If you compiled SystemDS from source, you can of course use the created JAR file with the run script.

``` bash
./bin/systemds path/to/the/SystemDS.jar hello.dml -args 10 10 1.0
```

### Running a script locally, in your SystemDS source environment

If you have cloned the SystemDS source repository and want to run your DML script with that, you can point the
shell script to the source directory by setting the `SYSTEMDS_ROOT` environment variable.

``` bash
SYSTEMDS_ROOT=../../code/my-systemds/source
./bin/systemds hello.dml -args 10 10 1.0
```

More about the environment setup can be found on : [running Systemds](http://apache.github.io/systemds/site/run).

### Running a script distributed on a Spark cluster

For running on a Spark cluster, the env variable SYSDS_DISTRIBUTED needs to be set (to something other than 0).
Per default, SystemDS will run in hybrid mode, pushing some instructions to the cluster and running others locally.
To force cluster mode in this little test, we will increase the matrix size to give the worker nodes in the cluster
something to do and force SystemDS to only generate Spark instructions by adding -exec spark to the command line
parameters:

``` bash
SYSDS_DISTRIBUTED=1
./bin/systemds hello.dml -args 10000 10000 1.0 -exec spark
```

The output should read something similar to this (the warning can be safely ignored):

``` bash
20/03/09 16:40:29 INFO api.DMLScript: BEGIN DML run 03/09/2020 16:40:29
20/03/09 16:40:30 WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Your hello world matrix contains:
250,902 207,246 305,621 182,892 234,394 258,587 132,225 259,684 255,774 228,338
296,740 261,475 379,148 156,640 304,543 267,684 191,867 258,826 373,633 276,497
215,877 186,171 332,165 201,091 289,179 265,160 125,890 289,836 320,434 287,394
389,057 332,681 336,182 285,432 310,218 340,838 301,308 354,130 410,698 282,453
325,903 264,745 377,086 242,436 277,836 285,519 190,167 358,228 332,295 288,034
360,858 301,739 398,514 265,299 333,124 321,178 240,755 299,871 428,856 300,128
368,983 291,729 303,091 191,586 231,050 280,335 266,906 278,203 395,130 203,706
173,610 114,076 157,683 140,927 145,605 145,654 143,674 192,044 196,735 166,428
310,329 258,840 286,302 231,136 305,804 300,016 266,434 297,557 392,566 281,211
249,234 196,488 216,662 180,294 165,482 169,318 172,686 204,275 296,595 148,888

SystemDS Statistics:
Total execution time:           0,122 sec.

20/03/09 16:40:30 INFO api.DMLScript: END DML run 03/09/2020 16:40:30
```

## Further reading

More documentation is available in the [SystemDS Homepage](https://systemds.apache.org/documentation)
