---
layout: global
title: Using SystemML with GPU
description: Using SystemML with GPU
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

# User Guide

To use SystemML on GPUs, please ensure that [CUDA 9](https://developer.nvidia.com/cuda-90-download-archive) and
[CuDNN 7](https://developer.nvidia.com/cudnn) is installed on your system.

## Python users

Please install SystemML using pip:
- For released version: `pip install systemml`
- For bleeding edge version: `pip install https://sparktc.ibmcloud.com/repo/latest/systemml-1.2.0-SNAPSHOT-python.tar.gz`

Then you can use the `setGPU(True)` method of [MLContext](http://apache.github.io/systemml/spark-mlcontext-programming-guide.html) and 
[MLLearn](http://apache.github.io/systemml/beginners-guide-python.html#invoke-systemmls-algorithms) APIs to enable the GPU usage.

```python
from systemml.mllearn import Caffe2DML
lenet = Caffe2DML(spark, solver='lenet_solver.proto', input_shape=(1, 28, 28))
lenet.setGPU(True)
```
To skip memory-checking and force all GPU-enabled operations on the GPU, please use the `setForceGPU(True)` method after `setGPU(True)` method.

```python
from systemml.mllearn import Caffe2DML
lenet = Caffe2DML(spark, solver='lenet_solver.proto', input_shape=(1, 28, 28))
lenet.setGPU(True).setForceGPU(True)
```

## Command-line users

To enable the GPU backend via command-line, please provide `systemml-1.*-extra.jar` in the classpath and `-gpu` flag.

```
spark-submit --jars systemml-1.*-extra.jar SystemML.jar -f myDML.dml -gpu
``` 

To skip memory-checking and force all GPU-enabled operations on the GPU, please provide `force` option to the `-gpu` flag.

```
spark-submit --jars systemml-1.*-extra.jar SystemML.jar -f myDML.dml -gpu force
``` 

## Scala users

To enable the GPU backend via command-line, please provide `systemml-1.*-extra.jar` in the classpath and use 
the `setGPU(True)` method of [MLContext](http://apache.github.io/systemml/spark-mlcontext-programming-guide.html) API to enable the GPU usage.

```
spark-shell --jars systemml-1.*-extra.jar,SystemML.jar
``` 

# Troubleshooting guide

- If you have older gcc (< 5.0) and if you get `libstdc++.so.6: version CXXABI_1.3.8 not found` error, please upgrade to gcc v5+. 
On Centos 5, you may have to compile gcc from the source:

```
sudo yum install libmpc-devel mpfr-devel gmp-devel zlib-devel*
curl ftp://ftp.gnu.org/pub/gnu/gcc/gcc-5.3.0/gcc-5.3.0.tar.bz2 -O
tar xvfj gcc-5.3.0.tar.bz2
cd gcc-5.3.0
./configure --with-system-zlib --disable-multilib --enable-languages=c,c++
num_cores=`grep -c ^processor /proc/cpuinfo`
make -j $num_cores
sudo make install
```