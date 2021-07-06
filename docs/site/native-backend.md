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
- [Linux](#linux)
- [Windows](#windows)
- [Command-line users](#command-line-users)
- [Scala Users](#scala-users)
- [Advanced Configuration](#advanced-configuration)
  - [Using single precision](#using-single-precision)
- [Training very deep network](#training-very-deep-network)
  - [Shadow buffer](#shadow-buffer)
  - [Unified memory allocator](#unified-memory-allocator)

## Requirements

### Hardware



### Software

Either of the following software is required to be installed in your system:

INTEL MKL

  Download [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html).
     Supported MKL version is `2017` to `2019.5`.

OpenBLAS
  
  Install [OpenBLAS](https://www.openblas.net/). Installation instructions on [GitHub](https://github.com/xianyi/OpenBLAS#installation-from-source).


### Add library to the System path

For Intel MKL
[Scripts to set environmental variables](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-linux-developer-guide/top/getting-started/setting-environment-variables/scripts-to-set-environment-variables.html)


## Linux



Note: All linux distributions may not support this. you might encounter some problems with driver
installations.

To check the CUDA compatible driver version:

Install [CUPTI](http://docs.nvidia.com/cuda/cupti/) which ships with CUDA toolkit for profiling.

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```

#### Installation check

```sh
$ nvidia-smi
```
