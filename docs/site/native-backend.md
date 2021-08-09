---
layout: site
title: Native Backend guide for SystemDS
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

This guide covers the Native BLAS optimizations and software setup for using SystemDS `native` mode.

- [Requirements](#requirements)
  - [Hardware](#hardware)
  - [Software](#software)
    - [INTEL MKL](#intel-mkl)
    - [OpenBLAS](#openblas)
- [Native BLAS setup](#native-blas-setup)
  - [Add library to the System path](#add-library-to-the-system-path)
  - [Enable Native BLAS in SystemDS](#enable-native-blas-in-systemds)
  - [Troubleshooting](#troubleshooting)


# Native BLAS mode

SystemDS implements all the matrix operations in Java. This simplifies deployment especially in
a distributed environment.

In Some cases (such as Deep Neural Networks), to take advantage of native BLAS instead of SystemDS
internal Java library for performing single node operations such as matrix multiplication, convolution etc.

By default, SystemDS will first attempt to use Intel MKL (if installed), and then OpenBLAS (if installed).
If none of the libraries are available, SystemDS falls back to its internal java library.

> Note: Current SystemDS version supported on **Linux**, **Windows** x86-64 platform.

## Requirements

### Hardware

To know Intel MKL system requirements, see
[Intel® oneAPI Math Kernel Library System Requirements](https://software.intel.com/content/www/us/en/develop/articles/oneapi-math-kernel-library-system-requirements.html)


### Software

Either of the following software is required to be installed in your system:

#### INTEL MKL

  Download [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html).
     Supported MKL version is `2017` to `2019.5`.

#### OpenBLAS
  
  Install [OpenBLAS](https://www.openblas.net/). Installation instructions on [GitHub](https://github.com/xianyi/OpenBLAS#installation-from-source).
  
  Note: In Ubuntu 20.04, remove `libopenblas0-pthread` package and install `libopenblas0-openmp` 
  instead. So that OpenBLAS will be installed with [OpenMP](https://www.openmp.org/) support.

## Native BLAS setup

### Add library to the System path

#### Intel MKL

[Scripts to set environmental variables](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-linux-developer-guide/top/getting-started/setting-environment-variables/scripts-to-set-environment-variables.html)

For example, in Linux to set Intel MKL libraries in the Path:

```sh
# path adjustments to that script according to your installation
source /opt/intel/bin/compilervars.sh intel64
```

#### OpenBLAS

Via commandline:

Java:

```sh
-Djava.library.path=/path/to/blas-n-other-dependencies
```

Note: This property can also be set with `sysds.native.blas.directory`.


### Enable Native BLAS in SystemDS

Set `sysds.native.blas property` to `mkl`, `openblas` as shown.

```xml
<!-- enables native blas for matrix multiplication and convolution, experimental feature (options: auto, mkl, openblas, none) -->
    <sysds.native.blas>mkl</sysds.native.blas>
```

### Troubleshooting

If there are issues loading libs because the gcc ABI changed from gcc 4.x to 5 and above.
In this case,
  - fiddle with the [patchelf](https://github.com/NixOS/patchelf) utility or
  - Compile a recent gcc and adjust your env vars (`PATH`, `LD_LIBRARY_PATH`)

