---
layout: site
title: Run SystemDS with GPU
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

This guide helps in installing pre-requisites for running with GPU and run instructions.

- [Requirements](#requirements)
- [Command-line users](#command-line-users)
- [Scala Users](#scala-users)
- [Advanced Configuration](#advanced-configuration)
  - [Using single precision](#using-single-precision)
- [Training very deep network](#training-very-deep-network)
  - [Shadow buffer](#shadow-buffer)
  - [Unified memory allocator](#unified-memory-allocator)

## Requirements

### Hardware

The following GPUs are supported:

* NVIDIA GPU cards with CUDA architectures 5.0, 6.0, 7.0, 7.5, 8.0 and higher than 8.0.
For CUDA enabled gpu cards at [CUDA GPUS](https://developer.nvidia.com/cuda-gpus)

* For GPUs with unsupported CUDA architectures, or to avoid JIT compilation from PTX, or to
use difference versions of the NVIDIA libraries, see the Linux build from source guide.

* Release artifacts contain PTX code for the latest supported CUDA architecture.

### Software

The following NVIDIA software is required to be installed in your system:

CUDA toolkit

  1. [NVIDIA GPU drivers](https://www.nvidia.com/drivers) - CUDA 10.2 requires >= 440.33 driver. see
     [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).
  3. [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
  4. [CUDNN 7.x](https://developer.nvidia.com/cudnn)


## Linux

One easiest way to install the NVIDIA software is with `apt` on Ubuntu. For other distributions
refer to the [CUDA install Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Note: All linux distributions may not support this. you might encounter some problems with driver
installations.

To check the CUDA compatible driver version:

```sh
CUDA_DRIVER_VERSION=$(apt-cache madison cuda-drivers | awk '{print $3}' | sort -r | while read line; do
   if dpkg --compare-versions $(dpkg-query -f='${Version}\n' -W nvidia-driver-${NVIDIA_DRIVER_VERSION}) ge $line ; then
       echo "$line"
       break
   fi
done)
```

Install [CUPTI](http://docs.nvidia.com/cuda/cupti/) which ships with CUDA toolkit for profiling.

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```

### Install CUDA with apt

The following instructions are for installing CUDA 10.2 on Ubuntu 18.04. These instructions
might work for other Debian-based distros.

Note: [Secure Boot](https://wiki.ubuntu.com/UEFI/SecureBoot) tends to complication installation.
These instructions may not address this.

#### Ubuntu 18.04 (CUDA 10.2)

```sh

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update

# get the machine-learning repo
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0
    
 # Reboot the system. And run `nvidia-smi` for GPU check.
```


## Command-line users

To enable the GPU backend via command-line, please provide `systemds-*-extra.jar` in the classpath and `-gpu` flag.

```
spark-submit --jars systemds-*-extra.jar SystemDS.jar -f myDML.dml -gpu
``` 

To skip memory-checking and force all GPU-enabled operations on the GPU, please provide `force` option to the `-gpu` flag.

```
spark-submit --jars systemds-*-extra.jar SystemDS.jar -f myDML.dml -gpu force
``` 

## Scala users

To enable the GPU backend via command-line, please provide `systemds-*-extra.jar` in the classpath and use 
the `setGPU(True)` method of MLContext API to enable the GPU usage.

```
spark-shell --jars systemds-*-extra.jar,SystemDS.jar
``` 

## Advanced Configuration

### Using single precision

By default, SystemDS uses double precision to store its matrices in the GPU memory.
To use single precision, the user needs to set the configuration property `sysds.floating.point.precision`
to `single`. However, with exception of BLAS operations, SystemDS always performs all CPU operations
in double precision.

### Training very deep network

#### Shadow buffer
To train very deep network with double precision, no additional configurations are necessary.
But to train very deep network with single precision, the user can speed up the eviction by 
using shadow buffer. The fraction of the driver memory to be allocated to the shadow buffer can  
be set by using the configuration property `sysds.gpu.eviction.shadow.bufferSize`.
In the current version, the shadow buffer is currently not guarded by SystemDS
and can potentially lead to OOM if the network is deep as well as wide.

#### Unified memory allocator

By default, SystemDS uses CUDA's memory allocator and performs on-demand eviction
using the eviction policy set by the configuration property `sysds.gpu.eviction.policy`.
To use CUDA's unified memory allocator that performs page-level eviction instead,
please set the configuration property `sysml.gpu.memory.allocator` to `unified_memory`.
