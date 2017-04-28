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

# User Guide

By default, SystemML implements all its matrix operations in Java.
This simplifies deployment especially in a distributed environment.

In some cases (such as deep learning), the user might want to use native BLAS
rather than SystemML's internal Java library for performing single-node
operations such matrix multiplication, convolution, etc.
By default, SystemML will first attempt to use Intel MKL (if installed)
and then OpenBLAS (if installed).
If both Intel MKL and OpenBLAS are not available, SystemML
falls back to its internal Java library.

To force SystemML to use internal Java library rather than native BLAS,
please set the configuration property `native.blas` to `false`.


## Step 1: Install BLAS

### Option 1: Install Intel MKL (recommended)

Download and install the [community version of Intel MKL](https://software.intel.com/sites/campaigns/nest/).
Intel requires you to first register your email address and then sends the download link to your email address
with license key.

* Linux/Mac users will have to extract the downloaded `.tgz` file, execute `install.sh` and follow the guided setup.
* Windows users will have to execute the downloaded `.exe` file and follow the guided setup.


### Option 2: Install OpenBLAS  

1. Linux:
	```bash
	# The default OpenBLAS (via yum/apt-get) uses its internal threading rather than OpenMP, 
	# which can lead to performance degradation when using SystemML. So, instead we recommend that you
	# compile OpenBLAS from the source. 
	# RedHat / CentOS: sudo yum install openblas
	# Ubuntu: sudo apt-get install openblas
	git clone https://github.com/xianyi/OpenBLAS.git
	cd OpenBLAS/
	make clean
	make USE_OPENMP=1
	sudo make install
	```

2. Windows:

We recommend that you build `libopenblas.dll` from source instead of 
downloading [pre-built binaries](https://sourceforge.net/projects/openblas/) to avoid performance degradation when using SystemML:

- Download source: `git clone https://github.com/xianyi/OpenBLAS.git`
- Install [CMake](https://cmake.org/download/) and Visual Studio Community Edition. 
- Open `CMake-GUI` and point source code to the openblas directory.
- Click `Add Entry` and add entry `USE_OPENMP` and select it.
- Then click `Configure`, `Generate` and then `Open Project`. This should open a Visual Studio IDE.
- Build libopenblas project and place the generated `libopenblas.dll` in your `PATH`.

3. Mac:
	
	```bash
	brew install openblas
	```

## Step 2: Install other dependencies

1. Linux:

	```bash
	# Centos/RedHat
	sudo yum install gcc-c++
	# Ubuntu
	sudo apt-get install g++ 
	```

We also depend on GNU OpenMP (gomp) which will be installed by GCC.
To find the location of `gomp` on your system, please use the command `ldconfig -p | grep libgomp`.
If gomp is available as `/lib64/libgomp.so.1` instead of `/lib64/libgomp.so`,
please add a softlink to it:

	```bash
	sudo ln -s /lib64/libgomp.so.1 /lib64/libgomp.so
	```


2. Windows:

To get other dependencies, either put below given dlls in your `PATH` or 
install Visual Studio Community edition to get these dlls:

- Visual C OpenMP: vcomp140.dll (for function: `omp_get_thread_num`).
- Visual C Runtime: vcruntime140.dll (for functions: `memcpy` and `memset`).
- API-MS-WIN-CRT-ENVIRONMENT-L1-1-0.DLL, API-MS-WIN-CRT-RUNTIME-L1-1-0.DLL and API-MS-WIN-CRT-HEAP-L1-1-0.DLL (for functions: `malloc` and `free`).

3. Mac:

	```bash
	brew install gcc --without-multilib
	```

# Developer Guide

This section describes how to compile shared libraries in the folder `src/main/cpp/lib`. 
This is required when the developer makes changes to cpp directory or while validating the source package during the release process.

To force SystemML to use OpenBLAS instead of Intel MKL if both are installed,
please set the environment variable `SYSTEMML_BLAS` to `openblas`.
This environment variable is used internally for testing and is not required for users.

## Intro to CMake
If you are familiar with cmake, skip this section.

In a regular project with a Makefile, the compiled object files are placed in the same directory as the source.
Sometimes we don't want to pollute the source tree. We might also want to have different binaries for different configurations. For instance, if we want to link a binary with separate libraries.
CMake supports out of source tree builds. As an illustration, you can create a directory called "BUILD" and invoke cmake like so : `cmake <path/to/source>`. The makefile and other config files are placed in this "BUILD" directory. You can now say `make` and the compiled objects and binary files are created in this directory. You can then create another "BUILD2" directory and repeat the process.
You can pass options to cmake as well. In this instance, it might be to specify whether to build with Intel MKL or OpenBLAS. This can be done from the command line with a "-D" appended to it, but more interestingly, it can also be done form a n-curses GUI which is invoked as `ccmake <path/to/source>`. (You may need to install this separately).
Also, the C, C++ compilers and their flags are picked up by cmake when set in standard environment variables. These are respectively `CC`, `CXX`, `CFLAGS` & `CXFLAGS`. As an example, they may be specified as:

	CXX=gcc-6 cmake ..

For this project, I typically make a directory in the `cpp` folder (this folder) and name it the config I use. For instance, `INTEL` for Intel MKL and `OPENBLAS` for OpenBLAS.

## 64-bit x86 Linux

1. Install `g++`, OpenBLAS and MKL using the above instructions

2. Set `JAVA_HOME` to JDK.

	export JAVA_HOME=<path to JDK 1.8>

3. Install cmake

	```bash
	# Centos/RedHat
	sudo yum install cmake3
	# Ubuntu
	sudo apt-get install cmake
	```

4. Compile the libs using the below script. 

	```bash
	mkdir INTEL && cd INTEL
	cmake -DUSE_INTEL_MKL=ON -DCMAKE_BUILD_TYPE=Release ..
	make install
	cd ..
	mkdir OPENBLAS && cd OPENBLAS
	cmake -DUSE_OPEN_BLAS=ON -DCMAKE_BUILD_TYPE=Release ..
	make install
	cd ..
	# The below script helps maintain this document as well as avoid accidental inclusion of non-standard dependencies.
	./check-dependency-linux-x86_64.sh
	```


The generated library files are placed in src/main/cpp/lib. This location can be changed from the CMakeLists.txt file.


The above script also validates whether additional dependencies have been added while compiling and warns the developer.  
The current set of dependencies other than MKL and OpenBLAS, are as follows:

- GNU Standard C++ Library: `libstdc++.so.6`
- GCC version 4.8 shared support library: `libgcc_s.so.1`
- The GNU libc libraries: `libm.so.6, libdl.so.2, libc.so.6, libpthread.so.0`
- GCC OpenMP v3.0 shared support library: `libgomp.so.1`
- Additional OpenBLAS dependencies: Fortran runtime (`libgfortran.so.3`) and GCC `__float128` shared support library (`libquadmath.so.0`)

If CMake cannot detect your OpenBLAS installation, set the `OpenBLAS_HOME` environment variable to the OpenBLAS Home.
	
## 64-bit x86 Windows

- Install MKL or Download the OpenBlas Binary
- Install Visual Studio Community Edition (tested on VS 2017)
- Use the CMake GUI, select the source directory, the output directory
- Press the `configure` button, set the `generator` and `use default native compilers` option
- Set the `CMAKE_BUILD_TYPE` to `Release`, this sets the appropriate optimization flags
- By default, `USE_INTEL_MKL` is selected, if you wanted to use OpenBLAS, unselect the `USE_INTEL_MKL`, select the `USE_OPEN_BLAS`.
- You might run into errors a couple of times, select the appropriate library and include files/directories (For MKL or OpenBLAS) a couple of times, and all the errors should go away.
- Then press generate. This will generate Visual Studio project files, which you can open in VS2017 to compile the libraries.

The current set of dependencies are as follows:
- MKL: mkl_rt.dll (for functions: `mkl_set_num_threads` and `cblas_dgemm`).
- OpenBLAS: libopenblas.dll (for functions: `openblas_set_num_threads` and `cblas_dgemm`).
- Visual C OpenMP: vcomp140.dll (for function: `omp_get_thread_num`).
- Visual C Runtime: vcruntime140.dll (for functions: `memcpy` and `memset`).
- API-MS-WIN-CRT-ENVIRONMENT-L1-1-0.DLL, API-MS-WIN-CRT-RUNTIME-L1-1-0.DLL and API-MS-WIN-CRT-HEAP-L1-1-0.DLL (for functions: `malloc` and `free`).
- KERNEL32.dll

If you get an error `Error LNK1181 cannot open input file 'C:/Program.obj'`, 
you may have to use quotation marks around the path in Visual Studio project properties.
- Property Pages > C/C++ > General > Additional Include Directories
- Property Pages > Linker > Command Line

If you get an error `CMake Error at cmake/FindOpenBLAS.cmake:71 (MESSAGE): Could not find OpenBLAS`,
please set the environment variable `OpenBLAS_HOME` or edit the variables `OpenBLAS_INCLUDE_DIR` and `OpenBLAS_LIB`
to point to the `include` directory and `libopenblas.dll.a` respectively.

If you get an error `install Library TARGETS given no DESTINATION!`, you can comment `install(TARGETS systemml preload LIBRARY DESTINATION lib)` 
in CMakeLists.txt and manually place the compiled dll in the `src/main/cpp/lib` directory.

## 64-bit x86 Mac

The version of clang that ships with Mac does not come with OpenMP. `brew install` either `llvm` or `g++`. The instructions that follow are for llvm:

1. Intel MKL - CMake should detect the MKL installation path, otherwise it can specified by the environment variable `MKLROOT`. To use (with clang):
```
mkdir INTEL && cd INTEL
CXX=/usr/local/opt/llvm/bin/clang++ CC=/usr/local/opt/llvm/bin/clang LDFLAGS=-L/usr/local/opt/llvm/lib CPPFLAGS=-I/usr/local/opt/llvm/include cmake  -DUSE_INTEL_MKL=ON -DCMAKE_BUILD_TYPE=Release ..
make install
```

### (with gcc-6):
```
mkdir INTEL && cd INTEL
CXX=g++-6 CC=gcc-6 cmake  -DUSE_INTEL_MKL=ON -DCMAKE_BUILD_TYPE=Release ..
make install
```

2. OpenBLAS - CMake should be able to detect the path of OpenBLAS. If it can't, set the `OpenBLAS` environment variable. If using `brew` to install OpenBLAS, set the `OpenBLAS_HOME` environment variable to `/usr/local/opt/openblas/`. To use (with clang):
```
export OpenBLAS_HOME=/usr/local/opt/openblas/
mkdir OPENBLAS && cd OPENBLAS
CXX=/usr/local/opt/llvm/bin/clang++ CC=/usr/local/opt/llvm/bin/clang LDFLAGS=-L/usr/local/opt/llvm/lib CPPFLAGS=-I/usr/local/opt/llvm/include cmake  -DUSE_OPEN_BLAS=ON -DCMAKE_BUILD_TYPE=Release ..
make install
```

### (with gcc-6):
```
export OpenBLAS_HOME=/usr/local/opt/openblas/
mkdir OPENBLAS && cd OPENBLAS
CXX=g++-6 CC=gcc-6 -DUSE_OPEN_BLAS=ON -DCMAKE_BUILD_TYPE=Release ..
make install
```