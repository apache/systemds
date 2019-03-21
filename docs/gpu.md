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

```
$ nvcc --version | grep release
Cuda compilation tools, release 9.0, V9.0.176
$ cat /usr/local/cuda/include/cudnn.h | grep "CUDNN_MAJOR\|CUDNN_MINOR"
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 0
#define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
```

Depending on the API, the GPU backend can be enabled in different way:

1. When invoking SystemML from command-line, the GPU backend can be enabled by providing the command-line `-gpu` flag.
2. When invoking SystemML using the (Python or Scala) MLContext and MLLearn (includes Caffe2DML and Keras2DML) APIs, please use the `setGPU(enable)` method.
3. When invoking SystemML using the JMLC API, please set the `useGpu` parameter in `org.apache.sysml.api.jmlc.Connection` class's `prepareScript` method.

Python users do not need to explicitly provide the jar during their invocation. 
For all other APIs, please remember to include the `systemml-*-extra.jar` in the classpath as described below.

## Command-line users

To enable the GPU backend via command-line, please provide `systemml-1.*-extra.jar` in the classpath and `-gpu` flag.

```
spark-submit --jars systemml-*-extra.jar SystemML.jar -f myDML.dml -gpu
``` 

To skip memory-checking and force all GPU-enabled operations on the GPU, please provide `force` option to the `-gpu` flag.

```
spark-submit --jars systemml-*-extra.jar SystemML.jar -f myDML.dml -gpu force
``` 

## Python users

Please install SystemML using pip:
- For released version: `pip install systemml`
- For bleeding edge version: 
```
git clone https://github.com/apache/systemml.git
cd systemml
mvn package -P distribution
pip install target/systemml-*-SNAPSHOT-python.tar.gz
```

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

## Scala users

To enable the GPU backend via command-line, please provide `systemml-*-extra.jar` in the classpath and use 
the `setGPU(True)` method of [MLContext](http://apache.github.io/systemml/spark-mlcontext-programming-guide.html) API to enable the GPU usage.

```
spark-shell --jars systemml-*-extra.jar,SystemML.jar
``` 

# Advanced Configuration

## Using single precision

By default, SystemML uses double precision to store its matrices in the GPU memory.
To use single precision, the user needs to set the configuration property 'sysml.floating.point.precision'
to 'single'. However, with exception of BLAS operations, SystemML always performs all CPU operations
in double precision.

## Training very deep network

### Shadow buffer
To train very deep network with double precision, no additional configurations are necessary.
But to train very deep network with single precision, the user can speed up the eviction by 
using shadow buffer. The fraction of the driver memory to be allocated to the shadow buffer can  
be set by using the configuration property 'sysml.gpu.eviction.shadow.bufferSize'.
In the current version, the shadow buffer is currently not guarded by SystemML
and can potentially lead to OOM if the network is deep as well as wide.

### Unified memory allocator

By default, SystemML uses CUDA's memory allocator and performs on-demand eviction
using the eviction policy set by the configuration property 'sysml.gpu.eviction.policy'.
To use CUDA's unified memory allocator that performs page-level eviction instead,
please set the configuration property 'sysml.gpu.memory.allocator' to 'unified_memory'.


# Frequently asked questions

### How do I find the CUDA and CuDNN version on my system?

- Make sure `/usr/local/cuda` is pointing to the right CUDA version.

```
ls -l /usr/local/cuda
```

- Get the CUDA version using `nvcc`

```
$ nvcc --version | grep release
Cuda compilation tools, release 9.0, V9.0.176
```

- Get the CuDNN version using the `cudnn.h` header file.

```
$ cat /usr/local/cuda/include/cudnn.h | grep "CUDNN_MAJOR\|CUDNN_MINOR"
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 0
#define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
```


### How do I verify the CUDA and CuDNN version that SystemML depends on?

- Check the `jcuda.version` property in SystemML's `pom.xml` file.
- Then find the CUDA dependency in [JCuda's documentation](http://www.jcuda.org/downloads/downloads.html).
- For you reference, here are the corresponding CUDA and CuDNN versions for given JCuda version:

| JCuda  | CUDA    | CuDNN |
|--------|---------|-------|
| 0.9.2  | 9.2     | 7.2   |
| 0.9.0d | 9.0.176 | 7.0.2 |
| 0.8.0  | 8.0.44  | 5.1   |


### How do I verify that CUDA is installed correctly?

- Make sure `/usr/local/cuda` is pointing to the right CUDA version.
- Make sure that `/usr/local/cuda/bin` are in your PATH.
```
$ nvcc --version
$ nvidia-smi 
```
- Make sure that `/usr/local/cuda/lib64` are in your `LD_LIBRARY_PATH`.
- Test using CUDA samples
```
$ cd /usr/local/cuda-9.0/samples/
$ sudo make
$ ./bin/x86_64/linux/release/deviceQuery
$ ./bin/x86_64/linux/release/bandwidthTest 
$ ./bin/x86_64/linux/release/matrixMulCUBLAS 
```

### How to install CUDA 9 on Centos 7 with yum?

```
sudo yum install cuda-9-0.x86_64
sudo ln -sfn /usr/local/cuda-9.0/ /usr/local/cuda
```

### What is the driver requirement for CUDA 9?

As per [Nvidia's documentation](https://docs.nvidia.com/deploy/cuda-compatibility/index.html), the drivers have to be `>= 384.81` version.

### What do I do if I get `CXXABI_1.3.8 not found` error?

If you have older gcc (< 5.0) and if you get `libstdc++.so.6: version CXXABI_1.3.8 not found` error, please upgrade to gcc v5+. 
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
