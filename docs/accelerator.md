---
layout: global
title: Using systemml-accelerator
description: Using systemml-accelerator
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

## Introduction

The [systemml-accelerator](https://github.com/niketanpansare/systemml-accelerator) packages system-dependent libraries
to simplify deployment. It allows SystemML to use native BLAS as well as hardware accelerators (such as Nvidia's GPU).

## Using native BLAS

By default, SystemML implements all its matrix operations in Java. This simplifies deployment especially in a distributed environment.
However, in some cases (such as deep learning), the user might want to use native BLAS rather than SystemML's internal Java library.
The current version supports only 64-bit JVM and Intel MKL and OpenBLAS. For any other setup, we fallback to SystemML's internal Java library.

### Steps for installing Intel MKL

Download and install the [community version of Intel MKL](https://software.intel.com/sites/campaigns/nest/). 
Intel requires you to first register your email address and then sends the download link to your email address with license key.

<div class="codetabs">
<div data-lang="Linux" markdown="1">
```bash
Extract the downloaded .tgz file and execute install.sh.
```
</div>
<div data-lang="Windows" markdown="1">
```bash
Execute the downloaded .exe file and follow the guided setup.
```
</div>
</div>

### Steps for installing OpenBLAS

<div class="codetabs">
<div data-lang="Linux" markdown="1">
```bash
# Fedora, Centos
sudo yum install openblas
# Ubuntu
sudo apt-get install openblas
```
</div>
<div data-lang="Windows" markdown="1">
```bash
Download the pre-built binaries or install from the source.
```
</div>
</div> 

Links:

1. [Pre-built OpenBLAS binaries](https://sourceforge.net/projects/openblas/), 

2. [OpenBLAS source](https://github.com/xianyi/OpenBLAS).

By default, SystemML searches for Intel MKL and then OpenBLAS to select the underlying BLAS.
If both are not found, we fallback to SystemML's internal Java library.
If you want to explicitly select the underlying BLAS or disable native BLAS, please set
the environment variable `SYSTEMML_BLAS` to `mkl or openblas or none` (default: `mkl`).

## Using GPU

SystemML requires that CUDA 8.0 and CuDNN 5.1 is installed on the machine to exploit GPU. 
If these libraries are not installed, we fall back to non-GPU plan.
Like native BLAS, to exploit GPU we require that `systemml-accelerator.jar` is available.

If you want to explicitly disable using GPU, please set the environment variable `SYSTEMML_GPU` to `none` (default: `cuda`).

To test if BLAS and/or GPU is enabled, please follow the below steps:

<div class="codetabs">
<div data-lang="PySpark" markdown="1">
```bash
$SPARK_HOME/bin/pyspark --jars systemml-accelerator.jar --jars systemml-accelerator.jar --driver-class-path systemml-accelerator.jar
>>> from systemml import random
>>> m1 = random.uniform(size=(1000,1000))
>>> m2 = random.uniform(size=(1000,1000))
>>> m3 = m1.dot(m2).toNumPy()
```
</div>
<div data-lang="Scala" markdown="1">
```bash
$SPARK_HOME/bin/spark-shell --jars systemml-accelerator.jar,SystemML.jar
scala> import org.apache.sysml.api.mlcontext._
scala> import org.apache.sysml.api.mlcontext.ScriptFactory._
scala> val ml = new MLContext(sc)
scala> val script = dml("X = matrix(0.1, rows=1000, cols=1000); Y = matrix(0.2, rows=1000, cols=1000); Z = X %*% Y; print(sum(Z))")
scala> ml.execute(script)
```
</div>
</div>

The above script should output either 

```bash
accelerator.BLASHelper: Found BLAS: (mkl/openblas)
or
accelerator.LibraryLoader: Unable to load (MKL/OpenBLAS)
```

Note: if `systemml-accelerator.jar` is not included via `--jars` (for spark-shell or spark-submit), then we fall back to SystemML's internal Java library.


## Frequently asked questions

1. How to find whether Native BLAS was available to SystemML ?

	Please check whether the logs contains following message
	 
	```bash
	INFO accelerator.BLASHelper: Successfully loaded systemml library with (openblas | mkl)
	```

2. How to find whether Native BLAS was used by SystemML for my DML script ?

	Please check the line `Number of Native Calls` in SystemML's statistics.
 
3. How to check if OpenBLAS or Intel MKL is installed on Linux ?

	```bash
	ldconfig -v -N | grep libopenblas
	
	ldconfig -v -N | grep libmkl_rt
	``` 

4. I have installed OpenBLAS on Linux, but it is not getting picked up. What do I do  ?

  - First check if you are using 64-bit Java.
  - Then, check is OpenBLAS is in the library path
  
    ```bash
    ldconfig -v -N | grep libopenblas
    ```
    
  - If the above commands points to a file `/lib64/libopenblas.so.0`, you may have to add a soft link
  
    ```bash
    sudo ln -s /lib64/libopenblas.so.0 /usr/lib64/libopenblas.so
    ```  

4. I have installed Intel MKL with default configuration. How do I ensure that it is available to SystemML ?

	On Linux/Mac, you can either add the library path to either `LD_LIBRARY_PATH` or pass it to java via `java.library.path`.
	
	```bash
	export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
	```
	
	On Windows, you can similarly add the library path 
	`C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\redist\intel64_win\mkl`
	to the environment variable `PATH` or pass it to java via `java.library.path`.

5. How to resolve the error message `OMP: Error #13: Assertion failure at kmp_csupport.c(538).` or `undefined symbol: omp_get_num_procs` ?

	The above error message suggests that GNU OpenMP required for MKL cannot be found. 
	To resolve this issue, please add a soft link to the GNU OpenMP shared library (i.e. libgomp):
	
	```bash
	ldconfig -v -N | grep libgomp
	
	sudo ln -s /lib64/libgomp.so.1 /usr/lib64/libgomp.so
	```
	
	Alternatively, you can preload the OpenMP library in your session by appending the following lines to your bash profile:
	
	```bash  
	export MKL_THREADING_LAYER=GNU
	
	export LD_PRELOAD=/usr/lib64/libgomp.so.1
	```
