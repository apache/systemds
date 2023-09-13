#!/bin/bash
#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

# This shell script compiles the required shared libraries for 64-bit Linux on x86 machine

# yum whatprovides libgcc_s.so.1
# GNU Standard C++ Library: libstdc++.so.6
# GCC version 4.8 shared support library: libgcc_s.so.1
# The GNU libc libraries: libm.so.6, libdl.so.2, libc.so.6, libpthread.so.0
# GCC OpenMP v3.0 shared support library: libgomp.so.1
gcc_toolkit="libgcc_s.so\|libm.so\|libstdc++\|libc.so\|libdl.so\|libgomp.so\|libpthread.so"
linux_loader="linux-vdso.so\|ld-linux-x86-64.so"
intel_mkl="libmkl_rt.so"

# Fortran runtime: libgfortran.so.3
# GCC __float128 shared support library: libquadmath.so.0
openblas="libopenblas.so\|libgfortran.so\|libquadmath.so"


if ! [ -x "$(command -v cmake)" ]; then
  echo 'Error: cmake is not installed.' >&2
  exit 1
fi

if ! [ -x "$(command -v patchelf)" ]; then
  echo 'Error: patchelf is not installed.' >&2
  exit 1
fi

# Check if Intel MKL is installed
if ! ldconfig -p | grep -q libmkl_rt; then
  echo "Intel MKL not found. Installing Intel MKL..."

  wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
  apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
  rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

  echo "deb https://apt.repos.intel.com/oneapi all main" |  tee /etc/apt/sources.list.d/oneAPI.list
  apt update
  apt install intel-basekit

  #set the env variables
  source /opt/intel/oneapi/setvars.sh

fi

# Check if OpenBLAS is installed
if ! ldconfig -p | grep -q libopenblas; then
  echo "OpenBLAS not found. Installing OpenBLAS..."

  apt-get update
  apt-get install libopenblas-dev -y
fi

# configure and compile INTEL MKL
cmake . -B INTEL -DUSE_INTEL_MKL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS="-DUSE_GNU_THREADING -m64"
cmake --build INTEL --target install --config Release
patchelf --add-needed libmkl_rt.so lib/libsystemds_mkl-Linux-x86_64.so
rm -R INTEL

# configure and compile OPENBLAS
cmake . -B OPENBLAS -DUSE_OPEN_BLAS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS="-m64"
cmake --build OPENBLAS --target install --config Release
patchelf --add-needed libopenblas.so.0 lib/libsystemds_openblas-Linux-x86_64.so
rm -R OPENBLAS

# check dependencies linux x86_64
echo "-----------------------------------------------------------------------"
echo "Check for unexpected dependencies added after code change or new setup:"
echo "Non-standard dependencies for libsystemds_mkl-linux-x86_64.so"
ldd lib/libsystemds_mkl-Linux-x86_64.so | grep -v $gcc_toolkit"\|$linux_loader\|"$intel_mkl
echo "Non-standard dependencies for libsystemds_openblas-linux-x86_64.so"
ldd lib/libsystemds_openblas-Linux-x86_64.so | grep -v $gcc_toolkit"\|$linux_loader\|"$openblas
echo "-----------------------------------------------------------------------"

# compile HE
cmake he/ -B HE -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build HE --target install --config Release
rm -R HE