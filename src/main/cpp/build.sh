#!/usr/bin/env bash
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

date

if ! [ -x "$(command -v cmake)" ]; then
  echo 'Error: cmake is not installed.' >&2
  exit 1
fi

if ! [ -x "$(command -v patchelf)" ]; then
  echo 'Error: patchelf is not installed.' >&2
  exit 1
fi

mkdir -p log

./build_mkl.sh > log/mkl_build.log  2>&1 & 
./build_BLAS.sh > log/BLAS_build.log 2>&1 &
./build_HE.sh > log/HE_build.log 2>&1 &
wait

if grep -q "Could NOT find MKL" log/mkl_build.log; then 
  echo "WARN: Missing MKL"
elif grep -q "Sucessfull install of MKL" log/mkl_build.log; then 
  echo "INFO: Sucessfull install of MKL"
else 
  cat log/mkl_build.log
fi

if grep -q "Could not find OpenBLAS lib" log/BLAS_build.log; then 
  echo "WARN: Missing OpenBLAS"
elif grep -q "Sucessfull install of OpenBLAS" log/BLAS_build.log; then 
  echo "INFO: Sucessfull install of OpenBLAS"
else 
  echo "ERROR: OpenBLAS install failed:"
  cat log/BLAS_build.log
fi

if grep -q 'Could not find a package configuration file provided by "SEAL"' log/HE_build.log; then
  echo "WARN: Missing SEAL install"
  echo "\
wget -qO- https://github.com/microsoft/SEAL/archive/refs/tags/v3.7.0.tar.gz | tar xzf - \
&& cd SEAL-3.7.0 \
&& cmake -S . -B build -DBUILD_SHARED_LIBS=ON \
&& cmake --build build \
&& cmake --install build"
elif grep -q "Sucessfull install of Homomorphic Encryption Liberary SEAL" log/HE_build.log; then 
  echo "INFO: Sucessfull install of HE SEAL"
else 
  cat log/HE_build.log
fi 

date